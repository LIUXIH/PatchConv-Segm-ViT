import argparse
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from dataloader import MyNyuDataloader
from nyu import NYUDataset
from metrics import Result, AverageMeter_fastdepth
import os
from PatchConvS_ASPP_SE_MViT import PatchConvSASPPSE_MViT
import time
import utils


modality_names = MyNyuDataloader.modality_names
data_names = ['nyudepthv2']
parser = argparse.ArgumentParser(description='Fine_tune')
parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                    choices=data_names,
                    help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                    help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str, metavar='N', help="gpu id")
parser.add_argument('--learning_rate', default=1e-4)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--resume', default='')
parser.add_argument('--maxdepth', default=50, type=int)
parser.set_defaults(cuda=True)

args = parser.parse_args()

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
              'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()


def main():
    print("creating data loaders...")
    root_val = '/../data/nyudepthv2/val'
    # root_test_real = 'E:/TransStructrue/real_data/row'
#   ======================dataload===================
    if args.data == 'nyudepthv2':
        val_dataset = NYUDataset(root_val, split='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.')
    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    cudnn.benchmark = True
    print("data loaders created.")

    model = PatchConvSASPPSE_MViT().cuda()
    # net_weights = model.state_dict()
#   ======================load model===================
    if args.resume != "":
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['model'])
        print('parameter loader successfully!')

    average_meter = AverageMeter_fastdepth()
    model.eval()
    end = time.time()
#   ======================inference===================
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        data_time = time.time() - end
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        gpu_time = time.time() - end
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'rgb':
            rgb = input

            if i == 0:
                img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = 'comparison_{}.png'.format(i)
                utils.save_image(img_merge, filename)

            # if result.rmse > val_rmse and result.rmse < val_rmse_m:
            #     img_merge = utils.merge_into_row(rgb, target, pred)
            #     filename = 'val_test_{}.png'.format(i)
            #     utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'     
          'MAE={average.mae:.3f}\n'       
          'Delta1={average.delta1:.3f}\n'   
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(average=avg, time=avg.gpu_time))


if __name__ == '__main__':
    main()
