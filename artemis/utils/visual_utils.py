import torch
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2mat
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_matrix
from utils.graphics_utils import matrix_to_quaternion
from gaussian_renderer import render
from PIL import Image
from scene.gaussian_model import GaussianParams
from utils.sh_utils import RGB2SH

SKIN_CMAP = plt.get_cmap("tab20")


def _rotate_around_center(params, spin_R, center=None):
    """rotate the Gaussians by spin_R (3x3) around center (default: their mean)"""
    params = GaussianParams(*params)
    spin_t = params.xyz.mean(0) if center is None else center
    spin_t = (torch.eye(3).cuda() - spin_R) @ spin_t[:, None]
    spin_T = torch.eye(4).cuda()
    spin_T[:3, :3] = spin_R
    spin_T[:3, 3] = spin_t.squeeze(-1)
    fr = quaternion_to_matrix(params.rotation)
    vis_xyz, vis_fr = transform_mu_frame(params.xyz, fr, spin_T)
    return params._replace(xyz=vis_xyz, rotation=matrix_to_quaternion(vis_fr))


def rotate_gaussian(params, spin_angle, up, center=None):
    """rotate around the axis up (the animal's vertical axis, view.aux['up']) through center (default: the Gaussians' mean), spin_angle in radians"""
    return _rotate_around_center(params, axis_angle_to_matrix(up.cuda().float() * spin_angle), center)


def rotate_gaussian_yaw_pitch(params, yaw, pitch):
    """rotate around the Gaussians' mean: pitch around X first, then yaw around Y (radians)"""
    return _rotate_around_center(params, torch.from_numpy(euler2mat(float(pitch), float(yaw), 0, "sxyz")).cuda().float())


def skin_weight_colors(params, gaussians, deform_model):
    """color each Gaussian by its skinning weights (canonical positions -> voxel LBS weights)"""
    sk_weights = deform_model.articulator.voxel_deformer(gaussians.get_xyz)
    sk_colors = torch.tensor(np.array(SKIN_CMAP.colors[:sk_weights.shape[1]]), dtype=torch.float32, device="cuda")
    params = GaussianParams(*params)
    skin_features = torch.zeros_like(params.features)
    skin_features[:, 0, :3] = RGB2SH(sk_weights @ sk_colors)
    return params._replace(features=skin_features)


def tensor_to_pil(image):
    """(C, H, W) tensor in [0, 1] -> uint8 PIL image"""
    image = torch.clamp(image.permute(1, 2, 0), 0.0, 1.0).detach().cpu().numpy()
    return Image.fromarray((image.squeeze() * 255).astype(np.uint8))


def save_gif(frames, path):
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=100, quality=50)


def render_spin_gif(view, params, active_sh_degree, max_sh_degree, pipeline, background, gif_path, n_spin=16, use_filter=False):
    """render the Gaussians spinning 360 degrees around the vertical axis"""
    frames = [tensor_to_pil(render(view, rotate_gaussian(params, 2 * np.pi * vid / n_spin, view.aux['up']), active_sh_degree, max_sh_degree,
                                   pipeline, background, use_filter=use_filter)["render"])
              for vid in range(n_spin)]
    save_gif(frames, gif_path)


def vis_render(gt_rgb, pred_rgbs, pred_dep, save_path=None):
    gt_rgb = gt_rgb.permute(1, 2, 0)
    n_pred = len(pred_rgbs.keys())
    n_vis = n_pred + 3
    pred_depth = pred_dep.squeeze(0)
    fig = plt.figure(figsize=(4 * n_vis, 4))
    plt.subplot(1, n_vis, 1)
    plt.imshow(torch.clamp(gt_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("GT"), plt.axis("off")
    keys = list(pred_rgbs.keys())
    if "deformed" in keys:
        error_k = "deformed"
    elif "refine_articulated" in keys:
        error_k = "refine_articulated"
    elif "canonical" in keys:
        error_k = "canonical"
    else:
        raise ValueError(f"error_k not in keys: {keys}")
    for i in range(n_pred):
        plt.subplot(1, n_vis, i+2)
        k = keys[i]
        pred_rgb = pred_rgbs[k].permute(1, 2, 0)
        plt.imshow(torch.clamp(pred_rgb, 0.0, 1.0).detach().cpu().numpy())
        plt.title(k), plt.axis("off")
        if k == error_k:
            error = torch.clamp(abs(pred_rgb - gt_rgb), 0.0, 1.0).detach().cpu().numpy().max(axis=-1)
    
    plt.subplot(1, n_vis, n_vis-1)
    cmap = plt.imshow(error)
    plt.title(f"Render Error ({error_k})"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)
    
    plt.subplot(1, n_vis, n_vis)
    depth = pred_depth.detach().cpu().numpy()
    cmap = plt.imshow(depth)
    plt.title("Pred Depth"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)


def vis_spin_view(view, params, active_sh_degree, max_sh_degree, pipeline, background, gif_path, n_spin=16):
    """posed Gaussians spinning in front of the camera"""
    render_spin_gif(view, params, active_sh_degree, max_sh_degree, pipeline, background, gif_path, n_spin)


def vis_canonical_view(view, params, active_sh_degree, max_sh_degree, pipeline, background, gif_path, n_spin=16):
    """canonical Gaussians spinning"""
    render_spin_gif(view, params, active_sh_degree, max_sh_degree, pipeline, background, gif_path, n_spin)


def transform_mu_frame(mu, frame, T):
    R, t = T[:3, :3], T[:3, 3]
    new_frame = torch.einsum("ij, njk->nik", R, frame)
    new_mu = torch.einsum("ij, nj->ni", R, mu) + t[None, :]
    return new_mu, new_frame
