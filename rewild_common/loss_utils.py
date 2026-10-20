#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)

def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, dice_weight=1.0):
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice_weight * dice


def weighted_directional_tv_loss(normals, weights, eps=1e-6):
    """
    Weighted directional TV loss on normals, ignoring the ±180° ambiguity of the normal direction.
    
    Args:
        normals: [B, 3, H, W] - unit normals
        weights: [B, 1, H, W] - weights (e.g. alpha)
        eps: avoids division by zero

    Returns:
        scalar loss
    """

    # similarity of neighboring normals along x
    dx = F.cosine_similarity(normals[:, :, :, :-1], normals[:, :, :, 1:], dim=1)  # [B, H, W-1]
    wx = weights[:, :, :, :-1] * weights[:, :, :, 1:]                             # [B, 1, H, W-1]

    # similarity of neighboring normals along y
    dy = F.cosine_similarity(normals[:, :, :-1, :], normals[:, :, 1:, :], dim=1)  # [B, H-1, W]
    wy = weights[:, :, :-1, :] * weights[:, :, 1:, :]                             # [B, 1, H-1, W]

    # loss = 1 - |cos|, sign-agnostic
    loss_x = ((1.0 - dx.abs()) * wx.squeeze(1)).sum() / (wx.sum() + eps)
    loss_y = ((1.0 - dy.abs()) * wy.squeeze(1)).sum() / (wy.sum() + eps)

    return loss_x + loss_y
