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
import numpy as np


##### camera conventions #####
# Cameras are estimated by Fauna (fauna/transforms_wflip_wRobust.json stores w2c, mvp and campos per frame).
# The renderer maps Gaussians with Fauna's mvp and gives the rasterizer an identity camera, so these
# constants belong together: change one and the renders no longer line up with the input frames.
FAUNA_FOV = 25 / 180 * np.pi   # FoV of Fauna's perspective projection (mvp) and of the rasterizer's tanfov
RASTER_FOV = 60 / 180 * np.pi  # FoV of the identity camera handed to the rasterizer (its projection matrix)
FLIP_X = np.diag(np.array([-1., 1., 1.])).astype(np.float32)  # mirror about the YoZ plane


def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0):
    #  right-handed, OpenGL style, -z forward
    y = np.tan(fovy / 2)
    return np.array([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=np.float32)


FAUNA_PROJ = perspective(FAUNA_FOV, 1, n=0.1, f=1000.0)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def _sqrt_positive_part(x):
    """sqrt(max(0, x)) with a zero subgradient where x <= 0, without boolean indexing (no host sync)"""
    positive = x > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, x, torch.ones_like(x))), torch.zeros_like(x))


def matrix_to_quaternion(matrix):
    """
    pytorch3d.transforms.matrix_to_quaternion (v0.7.8, BSD license) with a sync-free _sqrt_positive_part.
    matrix: (..., 3, 3) -> quaternions (..., 4), real part first.
    """
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))
    # the quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    # pick the best-conditioned candidate (largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    gather_indices = indices.unsqueeze(-1).expand(list(batch_dim) + [1, 4])
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return torch.where(out[..., 0:1] < 0, -out, out)  # real part >= 0
