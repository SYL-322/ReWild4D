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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    norm3Ds_precomp,
    extra_attrs,
    raster_settings,
):
    color, depth, norm, alpha, radii, extra = _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    )
    
    norm = torch.nn.functional.normalize(norm, p=2, dim=0)
    # 3, H, W
    return color, depth, norm, alpha, radii, extra

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    ):
        # restrict the length of extra attr values to avoid dynamically sized shared memory allocation
        assert extra_attrs.shape[0] == 0 or extra_attrs.shape[1] <= 34
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            norm3Ds_precomp,
            extra_attrs,
            extra_attrs.shape[1] if extra_attrs.shape[0] != 0 else 0,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, norm, alpha, radii, extra, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, norm, alpha, radii, extra, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, norm3Ds_precomp, radii, extra_attrs, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, depth, norm, alpha, radii, extra

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_norm, grad_out_alpha, _, grad_out_extra):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, norm3Ds_precomp, radii, extra_attrs, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                extra_attrs,
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                norm3Ds_precomp,
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_depth,
                grad_out_norm,
                grad_out_alpha,
                grad_out_extra,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_norm3Ds_precomp,
            grad_extra_attrs,
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

def precompute_covariance_and_normals(means, scales, rotations, campos, scale_modifier=1.0):
    """Differentiable equivalent of CUDA covariance and camera-facing normals.

    Rotations are activated unit quaternions (w, x, y, z), as on the CUDA path.
    Match its strict comparisons as well, including tied Gaussian scales.
    """
    w, x, y, z = rotations.unbind(-1)
    frame = torch.stack((1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                         2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                         2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)), dim=-1).reshape(-1, 3, 3)
    basis = frame * (scales * scale_modifier).unsqueeze(-2)
    covariance = basis @ basis.transpose(-1, -2)
    packed = torch.stack([covariance[:, i, j] for i, j in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]], dim=-1)
    sx, sy, sz = scales.unbind(-1)
    axis = torch.where((sx > sz) & (sy > sz), 2,
                       torch.where((sx > sy) & (sz > sy), 1, 0))
    normal = frame.gather(2, axis[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    normal = torch.where(((means - campos) * normal).sum(-1, keepdim=True) > 0, -normal, normal)
    return packed, normal


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3Ds_precomp = None, norm3Ds_precomp=None, extra_attrs=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if means3D.shape[0] == 0:
            # No CUDA tiles are dispatched for an empty scene. Return the
            # background explicitly, keeping zero gradients to empty inputs.
            zero = means3D.sum() * 0 + means2D.sum() * 0 + opacities.sum() * 0
            for value in [shs, colors_precomp, scales, rotations, cov3Ds_precomp, norm3Ds_precomp, extra_attrs]:
                if value is not None and value.requires_grad:
                    zero = zero + value.sum() * 0
            height, width = raster_settings.image_height, raster_settings.image_width
            image = raster_settings.bg[:, None, None].expand(-1, height, width) + zero
            scalar = means3D.new_zeros((1, height, width)) + zero
            normals = means3D.new_zeros((3, height, width)) + zero
            radii = torch.empty(0, dtype=torch.int32, device=means3D.device)
            extra = (means3D.new_zeros((extra_attrs.shape[-1], height, width)) + zero
                     if extra_attrs is not None and extra_attrs.shape[-1] else means3D.new_empty(0))
            return image, scalar, normals, scalar.clone(), radii, extra
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            if norm3Ds_precomp is None:
                raise ValueError('Precomputed covariance also requires precomputed normals')
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3Ds_precomp is None:
            cov3Ds_precomp = torch.Tensor([])
        if norm3Ds_precomp is None:
            norm3Ds_precomp = torch.Tensor([])
        if extra_attrs is None:
            extra_attrs = torch.Tensor([])
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3Ds_precomp,
            norm3Ds_precomp,
            extra_attrs,
            raster_settings, 
        )

