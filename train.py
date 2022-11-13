import argparse
import gc
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from dataloader import MyNyuDataloader
from nyu import NYUDataset
from utils import AverageMeter, ssim
from metrics import AverageMeter_fastdepth, Result
import os
import time
import datetime
from tensorboardX import SummaryWriter
from PatchConvS_ASPP_SE_MViT import PatchConvSASPPSE_MViT

modality_names = MyNyuDataloader.modality_names
data_names = ['nyudepthv2']
parser = argparse.ArgumentParser(description='Fine_tune')
parser.add_argument('--data', metavar='DATA', default='nyudepthv2',    # 将参数储存在“--data”的变量中 default为变量默认值
                    choices=data_names,
                    help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                    help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--gpu', default='0', type=str, metavar='N', help="gpu id")
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--resume', default='')
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.set_defaults(cuda=True)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def adjust_learning_rate_L(optimizer, epoch, learning_rate):

    if epoch >= 10 and epoch < 20:   #
        lr = learning_rate / 2
    elif epoch >= 20 and epoch <30:
        lr = learning_rate / 4
    elif epoch >= 30 and epoch < 40:               # 40
        lr = learning_rate / 8
    elif epoch >= 40 and epoch < 50:
        lr = learning_rate / 16
    elif epoch >= 50 and epoch <100:
        lr = learning_rate * 0.1
    elif epoch >= 100 and epoch <150:
        lr = learning_rate * 0.1 / 2
    elif epoch >= 150 and epoch <200:
        lr = learning_rate * 0.1 / 4
    elif epoch >= 200 and epoch <250:
        lr = learning_rate * 0.1 / 8
    elif epoch >= 250:
        lr = learning_rate * 0.1 / 16
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    print("creating data loaders...")
    root_train = '../data/nyudepthv2.tar/train'
    root_val = '../data/nyudepthv2.tar/val'

    if args.data == 'nyudepthv2':
        # train_dataset = MyNyuDataloader(root_train, split='train', modality=args.modality)
        train_dataset = NYUDataset(root_train, split='train', modality=args.modality)
        val_dataset = NYUDataset(root_val, split='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.')

    batch_size = 8
    prefix = 'depth_mixer2' + str(batch_size)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("data loaders created.")

    model = PatchConvSASPPSE_MViT().cuda()

#   ==============load pretrain weight=================
    if args.resume != "":
        state_dict = torch.load(args.resume)
        del_key = []
        for key, _ in state_dict['model'].items():
            if "up" in key:
                del_key.append(key)
        for key in del_key:
            del state_dict['model'][key]
        model.load_state_dict(state_dict['model'], strict=False)
        print('pretrain parameter loader successfully!')

    model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.learning_rate, args.epochs, batch_size), flush_secs=30)
    # l1 loss
    l1_criterion = nn.L1Loss()

#   ==============training=================
    for epoch in range(args.epochs):
        adjust_learning_rate_L(optimizer, epoch, args.learning_rate)
        running_loss = 0.0
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        model.train()
        c_time = time.time()

        for i, (input, target) in enumerate(train_loader):
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()

            output = model(input)
            mask = (target > 0).detach()
            # mask = target > 0.001

#           ==============loss function==============
            l_depth = l1_criterion(output[mask], target[mask]).cuda()
            # l_depth = F.smooth_l1_loss(output[mask], target[mask], reduction='mean').cuda()
            l_ssim = torch.clamp((1 - ssim(output, target))*0.5, 0, 1)
            loss = (0.1 * l_depth) + (1.0 * l_ssim)

            losses.update(loss.data.item(), input.size(0))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            batch_time.update(time.time() - c_time)
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

#           ==============print training information==============
            niter = epoch*N+i
            if i % 50 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                writer.add_scalar('Train/Loss', losses.val, niter)

        running_loss /= len(train_loader) / batch_size
        print('Epoch:', epoch + 1,
              'train_loss:', running_loss,
              'time:', round(time.time() - c_time, 3), 's')

        gc.collect()

        # testing
        test_average_meter = AverageMeter_fastdepth()
        model.eval()
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            test_date_time = time.time() - end

            # compute output
            end = time.time()
            with torch.no_grad():
                pred = model(input)
            gpu_time = time.time() - end

            # measure accuracy and record loss
            result = Result()
            result.evaluate(pred, target)
            test_average_meter.update(result, gpu_time, test_date_time, input.size(0))  # total metrics
            end = time.time()

            # save best rmse weight
        avg = test_average_meter.average()
        print('==========================')
        print('\n*\n'
                  'RMSE={average.rmse:.3f}\n'
                  'MAE={average.mae:.3f}\n'
                  'Delta1={average.delta1:.3f}\n'
                  'REL={average.absrel:.3f}\n'
                  'Lg10={average.lg10:.3f}\n'
                  't_GPU={time:.3f}\n'.format(
                   average=avg, time=avg.gpu_time))
        print('==========================')
        checkpoint_data = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_data, "XXX.pth.tar")


if __name__ == '__main__':
    main()
