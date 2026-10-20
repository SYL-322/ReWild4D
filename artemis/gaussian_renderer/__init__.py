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
import math
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer, precompute_covariance_and_normals
from utils.sh_utils import eval_sh


FILTER_SCALE_THRESHOLD = 0.0001  # use_filter: drop Gaussians whose largest scale is below this


def filter_gaussians_by_scale(means3D, scales, rotations, opacity, sh_features, threshold=FILTER_SCALE_THRESHOLD):
    """
    Filter Gaussians by the max axis of their scales and return the kept attributes.
    """
    max_scale = scales.max(dim=1).values  # shape: [N]

    keep_mask = max_scale >= threshold  # shape: [N], bool tensor

    means3D_filtered     = means3D[keep_mask]
    scales_filtered      = scales[keep_mask]
    rotations_filtered   = rotations[keep_mask]
    opacity_filtered     = opacity[keep_mask]
    sh_features_filtered = sh_features[keep_mask]

    return means3D_filtered, scales_filtered, rotations_filtered, opacity_filtered, sh_features_filtered


def render(viewpoint_camera, gaussians, active_sh_degree, max_sh_degree,
            pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, use_filter=False):
    """
    Render the scene with the Artemis camera of viewpoint_camera. The Gaussians live in the canonical frame of the
    animal and are mapped to the Artemis world with (x - v_center) * scale + c_center.
    Scale parameters are already in world units; Gaussian frames use Artemis axes.

    Background tensor (bg_color) must be on GPU!
    """
    means3D, scales, rotations, opacity, sh_features = gaussians
    if use_filter:
        if override_color is not None:
            override_color = override_color[scales.max(dim=1).values >= FILTER_SCALE_THRESHOLD]
        means3D, scales, rotations, opacity, sh_features = filter_gaussians_by_scale(
            means3D, scales, rotations, opacity, sh_features, threshold=FILTER_SCALE_THRESHOLD)

    # Retain screen-space position gradients for densification.
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    aux = viewpoint_camera.aux
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=aux['viewmatrix'],
        projmatrix=aux['viewmatrix'] @ aux['projmatrix'],
        sh_degree=active_sh_degree,
        campos=aux['campos'],
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = (means3D - aux['v_center']) * aux['scale'] + aux['c_center']
    
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = norm3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp, norm3D_precomp = precompute_covariance_and_normals(
            means3D, scales, rotations, aux['campos'], scaling_modifier)
        scales = rotations = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = override_color
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (max_sh_degree + 1) ** 2)
            dir_pp = (means3D - aux['campos'].repeat(sh_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = sh_features
    else:
        colors_precomp = override_color

    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp,
        norm3Ds_precomp = norm3D_precomp,
    )
    
    ret = {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "normal": rendered_norm,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    return ret
