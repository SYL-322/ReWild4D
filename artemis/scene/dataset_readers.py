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

# Artemis sequences: monocular videos taken from the multi-view Artemis captures (one camera per frame, circling the
# animal). Training uses the ground-truth Artemis cameras of artemis/transforms_artemis.json. The Fauna shape
# (fauna/prior.obj) is rotated into the Artemis canonical frame of the animal; the renderer maps it to the Artemis
# world with (x - v_center) * scale + c_center.

import os
from PIL import Image
from typing import NamedTuple, Optional
import numpy as np
import torch
import json
from pytorch3d.io import load_obj
from tqdm import trange
from camera_geometry import (artemis_rotation,
                             alignment_from_transforms, corrected_transforms)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    aux: Optional[dict] = None


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    obj_path: str
    skin_weights: np.array
    v_init: torch.Tensor


def artemis_up(path):
    """vertical axis of the animal in its Artemis canonical frame (Fauna up: +Y), for spinning visualizations"""
    return artemis_rotation(path) @ np.array([0, 1, 0], dtype=np.float32)


def load_prior(path):
    """Fauna shape rotated into the Artemis canonical frame, (N, 3) float32"""
    vts, _, _ = load_obj(os.path.join(path, "fauna/prior.obj"))
    return vts.numpy().astype(np.float32) @ artemis_rotation(path).T


def training_alignment(path):
    """Use the same canonical-to-world transform for training, evaluation and geometry."""
    with open(os.path.join(path, 'artemis/transforms_artemis.json')) as file:
        return alignment_from_transforms(json.load(file), load_prior(path))


def load_frame(path, file_path, resize_size):
    image_path = os.path.join(path, 'images', file_path)
    image = Image.open(image_path).convert("RGB").resize(resize_size)
    mask = Image.open(os.path.join(path, 'masks', file_path)).convert("L").resize(resize_size)
    return image, mask, image_path, os.path.splitext(file_path)[0]


def make_camera(uid, image, mask, image_path, image_name, fid, info, alignment, signal):
    """
    info: Artemis camera (viewmatrix, projmatrix, campos, FoVx, FoVy)
    signal = [m, v]
        m: camera group, 1 if the camera is derived from the original image's camera, -1 if from the flipped image's
        v: image, 1 for the original image, -1 for the flipped image
    """
    camera = {k: np.array(info[k]).astype(np.float32) for k in ['viewmatrix', 'projmatrix', 'campos']}
    return CameraInfo(uid=uid, R=np.eye(3), T=np.zeros(3), FovY=info['FoVy'], FovX=info['FoVx'], image=image, mask=mask,
                      image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=fid,
                      aux={**camera, **alignment, 'signal': np.array(signal).astype(np.float32)})


def readCamerasFromTransforms(path, transformsfile, use_flip=False, resize_size=(512, 512)):
    """
    Training cameras, as (original, flip) pairs of the same frame. Each pair holds an image with its Artemis camera,
    and the flipped image with the symmetric Artemis camera. Two series:
      original series: (original image, its camera)  + (flipped image, symmetric camera)              time t
      flip series:     (flipped image, symmetric camera) + (original image, its camera)   [use_flip]  time t
    """
    with open(os.path.join(path, transformsfile)) as file:
        contents = json.load(file)
    alignment = {**alignment_from_transforms(contents, load_prior(path)), 'up': artemis_up(path)}
    # Derive virtual cameras from the aligned Fauna bilateral plane.
    contents = corrected_transforms(contents, alignment, artemis_rotation(path))
    frames, frames_flip = contents["frames"], contents["frames_flip"]
    n_frame = len(frames)
    cam_infos = []
    for is_flip_series in ([False, True] if use_flip else [False]):
        for idx in trange(n_frame, desc="Loading frames"):
            info, info_flip = frames[idx], frames_flip[idx]
            assert info['file_path'] == info_flip['file_path']
            assert info['time'] == info_flip['time']

            image, mask, image_path, image_name = load_frame(path, info["file_path"], resize_size)
            image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_flip = mask.transpose(Image.FLIP_LEFT_RIGHT)
            t = (info['time']+1) / n_frame  # normalized source-frame time

            if is_flip_series:
                original_cam = make_camera(idx, image_flip, mask_flip, image_path, image_name, t, info_flip, alignment, [-1, -1])
                flip_cam = make_camera(idx, image, mask, image_path, image_name, t, info, alignment, [-1, 1])
            else:
                original_cam = make_camera(idx, image, mask, image_path, image_name, t, info, alignment, [1, 1])
                flip_cam = make_camera(idx, image_flip, mask_flip, image_path, image_name, t, info_flip, alignment, [1, -1])
            cam_infos.append({"original": original_cam, "flip": flip_cam})

    print(f"Loaded {len(cam_infos)} camera pairs.")
    return cam_infos


def readArtemisInfo(path, white_background, use_flip=False, resize_size=(512, 512)):
    skin_weights = torch.tensor(np.load(os.path.join(path, "fauna/W0.npy")))
    train_cam_infos = readCamerasFromTransforms(path, "artemis/transforms_artemis.json", use_flip, resize_size)
    # Input views for visualization; evaluation uses held-out views from artemis/val_info.json.
    test_cam_infos = [{"original": cam["original"]} for cam in train_cam_infos if cam["original"].aux['signal'][1] == 1]
    return SceneInfo(train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     obj_path=os.path.join(path, "fauna/prior.obj"),
                     v_init=torch.tensor(load_prior(path)).float(),
                     skin_weights=skin_weights)
