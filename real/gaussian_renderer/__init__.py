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
from utils.graphics_utils import FAUNA_FOV, matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_matrix


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
    Render the scene.

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

    tanfovx = math.tan(FAUNA_FOV * 0.5)
    tanfovy = math.tan(FAUNA_FOV * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    campos = viewpoint_camera.aux['campos']  # [3]
    mvp = viewpoint_camera.aux['mvp']  # [4, 4]
    means3D = means3D - campos
    means3D = torch.matmul(torch.nn.functional.pad(means3D[None], pad=(0,1), mode='constant', value=1.0), torch.transpose(mvp[None], 1, 2))[0,...,:3]
    R_global = viewpoint_camera.aux['Rot']
    # Gaussian frames map local axes to world coordinates. Apply the same
    # world-to-camera rotation on the left as for the Gaussian centers.
    rotations = matrix_to_quaternion(R_global @ quaternion_to_matrix(rotations))
    
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = norm3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp, norm3D_precomp = precompute_covariance_and_normals(
            means3D, scales, rotations, viewpoint_camera.camera_center, scaling_modifier)
        scales = rotations = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = override_color
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (max_sh_degree + 1) ** 2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
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
