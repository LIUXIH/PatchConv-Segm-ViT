import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import torch
from math import exp
import torch.nn.functional as F
from torch.optim import lr_scheduler
import logging
import os
# from pytorch3d.loss import chamfer_distance
# from torch.nn.utils.rnn import pad_sequence

cmap = plt.cm.viridis


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


# only pred   for test real data
def single_pred(input, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    depth_pred_col = colored_depthmap(depth_pred_cpu)
    img_merge = rgb
    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])   # np.hstack 数组堆叠

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])  # np.vstack 数组堆叠（沿第一个轴堆叠）


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, win_size=11, window=None, size_average=True, full=False):
    # L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(win_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.cuda())

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        # print("mu1_sq:", mu1_sq, mu2_sq, mu1_mu2)

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
        # print("sigma_sq:", sigma1_sq, sigma2_sq, sigma12)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 /v2)

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        # print(ssim_map.shape)

        if size_average:
            ret = ssim_map.mean()
            # print(ret)
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret


import torch.nn as nn
import torch


class MonodepthLoss(nn.modules.Module):
    def __init__(self, SSIM_w = 0.85):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        # print(x.shape)
        # print(y.shape)

        mu_x = nn.AvgPool2d(kernel_size=3, stride=1)(x)
        mu_y = nn.AvgPool2d(kernel_size=3, stride=1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(kernel_size=3, stride=1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(kernel_size=3, stride=1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(kernel_size=3, stride=1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq +C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def forward(self, output, target):
        ssim_ = torch.mean(self.SSIM(output, target))
        # L1_loss = torch.mean(torch.abs(output - target))
        # loss = self.SSIM_w * ssim_ + (1 - self.SSIM_w) * L1_loss
        loss = ssim_
        return loss


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)   # StepLR 等间隔调整学习率
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def update_learning_rate(optimizers, scheduler):
    scheduler.step()
    lr = optimizers.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

    return lr


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


# class BinsChamferLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.name = "ChamferLoss"
#
#     def _slow_forward(self, bins, target_depth_maps):
#         bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
#         n, p = bin_centers.shape
#         input_points = bin_centers.veiw(n, p, 1)
#         target_points = target_depth_maps.flatten(1)
#         mask = target_points.ge(1e-3)
#         target_points = [p[m] for p, m in zip(target_points, mask)]
#         target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
#         target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)
#         loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
#         return loss
