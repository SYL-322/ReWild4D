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

import os
from PIL import Image
from typing import NamedTuple, Optional
import numpy as np
import torch
import json
from pytorch3d.io import load_obj
from tqdm import trange, tqdm
from utils.graphics_utils import FAUNA_PROJ, RASTER_FOV, FLIP_X

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
    frame_time_step: float

def load_fauna_json(path, transformsfile):
    """per-frame Fauna cameras / poses of the original frames and of the left-right flipped frames"""
    info_path = os.path.join(path, transformsfile)
    fauna_dir = os.path.dirname(info_path)
    contents = json.load(open(info_path, 'r'))
    poses = np.load(os.path.join(fauna_dir, "poses.npy"))
    poses_flip = np.load(os.path.join(fauna_dir, "poses_flip.npy"))
    return contents["frames"], contents['frames_flip'], poses, poses_flip


def load_frame(path, file_path, resize_size):
    image_path = os.path.join(path, 'images', file_path)
    image = Image.open(image_path).convert("RGB").resize(resize_size)
    mask = Image.open(os.path.join(path, 'masks', file_path)).convert("L").resize(resize_size)
    return image, mask, image_path, os.path.splitext(file_path)[0]


def json_camera(info):
    """camera estimated by Fauna for this image"""
    w2c = np.array(info['w2c'][0]).astype(np.float32)
    return {'Rot': w2c[:3, :3], 'Trans': w2c[:3, 3],
            'mvp': np.array(info['mvp'][0]).astype(np.float32),
            'campos': np.array(info['campos'][0]).astype(np.float32)}


def mirrored_camera(info):
    """camera mirrored about the YoZ plane: it sees the left-right flipped image of `info`"""
    w2c = np.array(info['w2c'][0]).astype(np.float32)
    R = FLIP_X @ w2c[:3, :3] @ FLIP_X
    T = FLIP_X @ w2c[:3, 3]
    w2c_sym = np.eye(4).astype(np.float32)
    w2c_sym[:3, :3] = R
    w2c_sym[:3, 3] = T
    return {'Rot': R, 'Trans': T, 'mvp': FAUNA_PROJ @ w2c_sym, 'campos': -R.T @ T}


def make_camera(uid, image, mask, image_path, image_name, fid, camera, pose, signal):
    """
    signal = [m, v]
        m: camera group, 1 if the camera is derived from the original image's camera, -1 if from the flipped image's
        v: image, 1 for the original image, -1 for the flipped image
    """
    return CameraInfo(uid=uid, R=np.eye(3), T=np.zeros(3), FovY=RASTER_FOV, FovX=RASTER_FOV, image=image, mask=mask,
                      image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=fid,
                      aux={**camera, 'pose': pose, 'signal': np.array(signal).astype(np.float32)})


def readCamerasFromTransforms(path, transformsfile, use_flip=False, resize_size=(512, 512), load_all=False):
    """
    Training cameras, as (original, flip) pairs of the same frame. Each pair holds an image, the camera Fauna
    estimated for it, and the mirrored camera that sees the flipped image. Two series:
      original series: (original image, its camera)  + (flipped image, mirrored camera)
      flip series:     (flipped image, its camera)   + (original image, mirrored camera)   [use_flip]
    Each camera series uses its own is_robust flag; validation frames are skipped unless load_all.
    """
    frames, frames_flip, poses, poses_flip = load_fauna_json(path, transformsfile)
    n_frame = len(frames)
    cam_infos = []
    for is_flip_series in ([False, True] if use_flip else [False]):
        for idx in trange(n_frame, desc="Loading frames"):
            info, info_flip = frames[idx], frames_flip[idx]
            selected_info = info_flip if is_flip_series else info
            if not selected_info['is_robust']:
                continue
            if info['is_val'] and not load_all:
                continue
            assert info['file_path'] == info_flip['file_path']
            assert info['time'] == info_flip['time']
            
            image, mask, image_path, image_name = load_frame(path, info["file_path"], resize_size)
            image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_flip = mask.transpose(Image.FLIP_LEFT_RIGHT)
            pose, pose_flip = poses[idx].astype(np.float32), poses_flip[idx].astype(np.float32)
            t = (idx+1) / n_frame  # time in (0, 1], for both original and flip views
            
            if is_flip_series:
                original_cam = make_camera(idx, image_flip, mask_flip, image_path, image_name, t, json_camera(info_flip), pose_flip, [-1, -1])
                flip_cam = make_camera(idx, image, mask, image_path, image_name, t, mirrored_camera(info_flip), pose, [-1, 1])
            else:
                original_cam = make_camera(idx, image, mask, image_path, image_name, t, json_camera(info), pose, [1, 1])
                flip_cam = make_camera(idx, image_flip, mask_flip, image_path, image_name, t, mirrored_camera(info), pose_flip, [1, -1])
            cam_infos.append({"original": original_cam, "flip": flip_cam})

    print(f"Loaded {len(cam_infos)} camera pairs.")
    return cam_infos


def readValCamerasFromTransforms(path, transformsfile, resize_size=(512, 512), load_all=False):
    """
    Evaluation cameras of the original images: validation frames (all frames if load_all) that are robust in the
    original or the flipped image. If only the flipped image is robust, its mirrored camera is used.
    """
    frames, frames_flip, poses, _ = load_fauna_json(path, transformsfile)
    n_frame = len(frames)
    cam_infos = []
    for idx in tqdm(range(n_frame), desc="Loading frames"):
        info, info_flip = frames[idx], frames_flip[idx]
        if not info['is_val'] and not load_all:
            continue
        if not info['is_robust'] and not info_flip['is_robust']:
            continue
        
        image, mask, image_path, image_name = load_frame(path, info["file_path"], resize_size)
        pose = poses[idx].astype(np.float32)
        t = (idx+1) / n_frame  # time in (0, 1], for both original and flip views
        if info['is_robust']:
            cam = make_camera(idx, image, mask, image_path, image_name, t, json_camera(info), pose, [1, 1])
        else:
            cam = make_camera(idx, image, mask, image_path, image_name, t, mirrored_camera(info_flip), pose, [-1, 1])
        cam_infos.append({"original": cam})
            
    print(f"Loaded {len(cam_infos)} cameras for eval.")
    return cam_infos
     
    
def readFaunaInfo(path, white_background, use_flip=False, resize_size=(512, 512), load_all=False, train_on_val=False):
    pts_path = os.path.join(path, "fauna/prior.obj")
    v_init, _, _ = load_obj(pts_path)
    
    skin_weights_path = os.path.join(path, "fauna/W0.npy")
    skin_weights = np.load(skin_weights_path)
    skin_weights = torch.tensor(skin_weights)
    
    test_cam_infos = readValCamerasFromTransforms(
        path, "fauna/transforms_wflip_wRobust.json", resize_size, load_all=load_all)
    assert len(test_cam_infos) != 0, "does not have test cameras in the current setup."

    train_cam_infos = readCamerasFromTransforms(
        path, "fauna/transforms_wflip_wRobust.json", use_flip, resize_size, load_all=train_on_val)

    # Camera times are (idx + 1) / N, before validation / robustness filtering.
    # Keep the same frame unit regardless of held-out frames or flip augmentation.
    with open(os.path.join(path, "fauna/transforms_wflip_wRobust.json")) as stream:
        frame_time_step = 1.0 / len(json.load(stream)["frames"])
    
    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           obj_path=pts_path,
                           v_init=v_init,
                           skin_weights=skin_weights,
                           frame_time_step=frame_time_step)
    return scene_info
    
    
sceneLoadTypeCallbacks = {
    "Fauna": readFaunaInfo,  # data processed by Fauna
}
