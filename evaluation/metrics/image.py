"""PSNR / SSIM / LPIPS with dataset-specific decoding and backgrounds."""
import os
from glob import glob
import numpy as np
import torch
from PIL import Image
import piq

device = 'cuda'
_lpips_alex = None

def compute_psnr(img1, img2):
    '''
    img1: (N, C, H, W)
    img2: (N, C, H, W)
    '''
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean().item()

def compute_ssim(img1, img2):
    '''
    img1: (N, C, H, W)
    img2: (N, C, H, W)
    '''
    return piq.ssim(img1, img2, data_range=1.0).item()

def compute_lpips_alex(img1, img2):
    '''
    img1: (N, C, H, W) in [-1, 1]
    img2: (N, C, H, W) in [-1, 1]
    '''
    global _lpips_alex
    if _lpips_alex is None:
        import lpips
        _lpips_alex = lpips.LPIPS(net='alex').to(device)
    return _lpips_alex(img1, img2).mean().item()

def rgba_to_rgb_with_background(img_rgba, bg_color=(255, 255, 255)):
    background = Image.new("RGBA", img_rgba.size, bg_color + (255,))
    return Image.alpha_composite(background, img_rgba).convert("RGB")

def load_rgba_images(pattern, bg_color=(0, 0, 0)):
    paths = sorted(glob(os.fspath(pattern))) if isinstance(pattern, (str, os.PathLike)) else pattern
    pils = [rgba_to_rgb_with_background(Image.open(f), bg_color=bg_color) for f in paths]
    return torch.stack([torch.tensor(np.array(img.convert('RGB')) / 255.0).permute(2, 0, 1) for img in pils], dim=0)

def eval_input_views(eval_dir, source_path=None):
    """eval_dir/original_GT/*_gt_rgba.png vs eval_dir/original_pred/*_pred_rgba.png"""
    from evaluation.protocols.real_frames import matched_input_paths
    pred_paths, gt_paths = matched_input_paths(eval_dir, source_path)
    batch_gt_img = load_rgba_images(gt_paths)
    batch_pred_img = load_rgba_images(pred_paths)
    return {
        'psnr': compute_psnr(batch_gt_img, batch_pred_img),
        'ssim': compute_ssim(batch_gt_img, batch_pred_img),
        'lpips_alex': compute_lpips_alex(batch_gt_img.float().cuda() * 2 - 1, batch_pred_img.float().cuda() * 2 - 1),
    }

def load_rgb_images(paths):
    return torch.stack([torch.tensor(np.array(Image.open(f).convert('RGB')) / 255.0).permute(2, 0, 1) for f in paths], dim=0)

def eval_multiview(eval_dir, gt_dir, batch_size=32, *, val_names):
    """
    eval_dir/<frame>/<view>.jpg vs gt_dir/images/<frame>/<view>.jpg (Artemis novel views), averaged over all
    images (per-batch means weighted by the batch size)
    """
    from evaluation.protocols.artemis_views import matched_view_paths
    pred_paths, gt_paths = matched_view_paths(eval_dir, gt_dir, val_names)

    sums = {'psnr': 0.0, 'ssim': 0.0, 'lpips_alex': 0.0}
    for i in range(0, len(pred_paths), batch_size):
        pred = load_rgb_images(pred_paths[i:i+batch_size]).cuda()
        gt = load_rgb_images(gt_paths[i:i+batch_size]).cuda()
        sums['psnr'] += compute_psnr(gt, pred) * len(gt)
        sums['ssim'] += compute_ssim(gt, pred) * len(gt)
        sums['lpips_alex'] += compute_lpips_alex(gt.float() * 2 - 1, pred.float() * 2 - 1) * len(gt)
    return {k: v / len(pred_paths) for k, v in sums.items()}
