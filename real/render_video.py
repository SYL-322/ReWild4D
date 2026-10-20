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
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, prepare_models
from utils.visual_utils import (rotate_gaussian, rotate_gaussian_yaw_pitch, shift_by_campos, skin_weight_colors,
                                tensor_to_pil, save_gif, render_spin_gif)

N_FRAMES = 64  # gif length; short sequences are padded back and forth


def round_pad_list(input_list, total_length):
    """pad to total_length by playing the list forward and backward"""
    output_list = []
    for i in range(int(np.ceil(total_length / len(input_list)))):
        output_list.extend(input_list if i % 2 == 0 else reversed(input_list))
    return output_list[:total_length]


def _render_spinning_video(model_path, name, iteration, cam_views, gaussians, pipeline, background, deform_model,
                           rotate_fn, tag, use_filter, render_skin, spin_frame_ids=()):
    """
    One gif per stage (deformed: "spin", articulated: "spin0") in which time runs forward and the view rotates
    with rotate_fn(params, frame_index). Frames in spin_frame_ids also get a 32-view spin gif.
    """
    if 'train' in name:
        cam_views = [view for view in cam_views if view['original'].aux['signal'][1] == 1]  # original image series only
    render_path = os.path.join(model_path, name, f"ours_{iteration}{'_skin' if render_skin else ''}", "renders")
    os.makedirs(render_path, exist_ok=True)
    deform_model.update_status(0)
    sh = (gaussians.active_sh_degree, gaussians.max_sh_degree)
    
    frame_paths = {'': [], '0': []}
    for idx, view_dict in enumerate(tqdm(cam_views, desc=f"Rendering {name}{tag}")):
        view = view_dict["original"]
        view.load2device()
        stages, _ = deform_model.train_step(gaussians, view.aux['pose'].cuda(), view.fid.cuda().unsqueeze(0), signal=view.aux['signal'])
        if render_skin:
            stages = [skin_weight_colors(stage, gaussians, deform_model) for stage in stages]
        
        for suffix, stage in [('', stages[-1]), ('0', stages[0])]:
            centered = shift_by_campos(stage, view)
            if idx in spin_frame_ids:
                render_spin_gif(view, centered, *sh, pipeline, background, render_path + f'_spin{suffix}_{view.image_name}.gif',
                                n_spin=32, use_filter=use_filter)
            image = render(view, rotate_fn(centered, idx % N_FRAMES), *sh, pipeline, background, use_filter=use_filter)["render"]
            path = os.path.join(render_path, f'spin{suffix}{tag}_{view.image_name}.jpg')
            tensor_to_pil(image).save(path)
            frame_paths[suffix].append(path)
    
    for suffix, paths in frame_paths.items():
        if len(paths) < N_FRAMES:
            paths = round_pad_list(paths, N_FRAMES)
        save_gif([Image.open(p) for p in paths], render_path + f'_spin{suffix}{tag}.gif')
    shutil.rmtree(render_path)


def render_set_spin(model_path, name, iteration, cam_views, gaussians, pipeline, background, deform_model,
                    save_all_spin=True, use_filter=False, render_skin=False):
    """spin around the vertical axis; with save_all_spin, also a 32-view gif for 6 (or all, if <= 10) frames"""
    n = len(cam_views) if 'train' not in name else sum(v['original'].aux['signal'][1] == 1 for v in cam_views)
    spin_frame_ids = []
    if save_all_spin:
        spin_frame_ids = torch.linspace(0, n - 1, 6).long().tolist() if n > 10 else list(range(n))
    _render_spinning_video(model_path, name, iteration, cam_views, gaussians, pipeline, background, deform_model,
                           lambda params, i: rotate_gaussian(params, 2 * np.pi * i / N_FRAMES), "",
                           use_filter, render_skin, spin_frame_ids)


def render_set_spin_yaw_pitch(model_path, name, iteration, cam_views, gaussians, pipeline, background, deform_model,
                              use_filter=False, render_skin=False):
    """full yaw turn starting at pi/2 while the pitch oscillates as 0.25 + 0.5 * sin(t)"""
    yaws = np.linspace(0 + np.pi / 2.0, 2 * np.pi + np.pi / 2.0, N_FRAMES, endpoint=False)
    pitches = 0.25 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False))
    _render_spinning_video(model_path, name, iteration, cam_views, gaussians, pipeline, background, deform_model,
                           lambda params, i: rotate_gaussian_yaw_pitch(params, yaws[i], pitches[i]), "_yaw_pitch",
                           use_filter, render_skin)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians, deform_model = prepare_models(dataset, iteration)
        scene = Scene(dataset, gaussians, load_iteration=iteration, load_flip=False, load_all=True)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # spinning views of the posed Gaussians, colored by RGB and by skinning weights
        for render_skin in [False, True]:
            render_set_spin(dataset.model_path, "vis_spin", scene.loaded_iter, scene.getTestCameras(),
                            gaussians, pipeline, background, deform_model,
                            save_all_spin=True, use_filter=True, render_skin=render_skin)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render spinning-view gifs of a trained model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
