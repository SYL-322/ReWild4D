"""Render 4D GIFs from val_info camera 0 as time advances.

The animal rotates around its canonical up axis once every 64 frames.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import prepare_models
from camera_geometry import artemis_animal
from scene.dataset_readers import training_alignment
from utils.general_utils import set_random_seed
from utils.visual_utils import rotate_gaussian, skin_weight_colors, tensor_to_pil, save_gif


class FixedArtemisView:
    """Load camera 0 from val_info and alignment from the input cameras."""
    def __init__(self, val_info):
        camera = val_info['camera']['0']
        self.FoVx = float(np.float32(camera['FoVx']))
        self.FoVy = float(np.float32(camera['FoVy']))
        self.image_height = self.image_width = 512
        self.aux = {key: torch.tensor(camera[key], dtype=torch.float32, device='cuda')
                    for key in ['viewmatrix', 'projmatrix', 'campos']}
        self.aux.update({key: torch.tensor(val_info[key], dtype=torch.float32, device='cuda')
                         for key in ['v_center', 'c_center', 'scale']})


def animation_frame_ids(val_info, n_spinning=64):
    """Keep original source timestamps; pad short motions forward and backward."""
    frame_ids = [int(frame) for frame in val_info['val_names']]
    if not frame_ids:
        raise ValueError('val_info contains no frames')
    if len(frame_ids) >= n_spinning:
        return frame_ids
    padded = []
    while len(padded) < n_spinning:
        padded.extend(frame_ids)
        padded.extend(reversed(frame_ids))
    return padded[:n_spinning]


def spin_axis(source_path):
    """Visualization spin axis: +Z for duck/bear, +Y for other animals."""
    return [0., 0., 1.] if artemis_animal(source_path) in ['duck', 'bear'] else [0., 1., 0.]


@torch.no_grad()
def render_4d_results(dataset, iteration, pipeline, output_dir=None, render_skin=True):
    gaussians, deform_model = prepare_models(dataset, iteration)
    deform_model.update_status(0)
    val_info = json.loads((Path(dataset.source_path) / 'artemis/val_info.json').read_text())
    val_info.update(training_alignment(dataset.source_path))
    view = FixedArtemisView(val_info)
    up = torch.tensor(spin_axis(dataset.source_path), dtype=torch.float32, device='cuda')
    background = torch.ones(3, dtype=torch.float32, device='cuda')
    signal = torch.ones(2, dtype=torch.float32, device='cuda')
    output = Path(output_dir) if output_dir else Path(dataset.model_path) / f'eval_{iteration}_gifs'
    output.mkdir(parents=True, exist_ok=True)
    frames = {'4d': [], '4d_stage0': []}
    if render_skin:
        frames.update({'4d_skin': [], '4d_skin_stage0': []})
    num_source_frames = len(val_info['val_names'])
    for index, frame_id in enumerate(tqdm(animation_frame_ids(val_info), desc='Rendering fixed-camera 4D')):
        time = torch.tensor([[(frame_id + 1) / num_source_frames]], dtype=torch.float32, device='cuda')
        stages, _ = deform_model.train_step(gaussians, time, signal=signal)
        for name, stage in [('4d', stages[-1]), ('4d_stage0', stages[0])]:
            variants = [(name, stage)]
            if render_skin:
                skin_name = '4d_skin' if name == '4d' else '4d_skin_stage0'
                variants.append((skin_name, skin_weight_colors(stage, gaussians, deform_model)))
            for tag, params in variants:
                params = rotate_gaussian(params, 2 * np.pi * index / 64, up)
                image = render(view, params, gaussians.active_sh_degree, gaussians.max_sh_degree,
                               pipeline, background, use_filter=True)['render']
                frames[tag].append(tensor_to_pil(image))
    for tag, images in frames.items():
        save_gif(images, str(output / f'{tag}.gif'))
    (output / 'render_metadata.json').write_text(json.dumps({
        'protocol': 'fixed_camera_4d', 'camera_id': 0, 'up': up.cpu().tolist(),
        'spin_frames_per_turn': 64, 'frame_ids': animation_frame_ids(val_info),
        'source_frame_count': num_source_frames, 'white_background': True, 'scale_filter': True,
        'model_path': dataset.model_path, 'iteration': iteration,
    }, indent=2))
    print(f'Saved fixed-camera 4D GIFs to {output}')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--skip_skin', action='store_true')
    args = get_combined_args(parser)
    set_random_seed(0)
    render_4d_results(model.extract(args), args.iteration, pipeline.extract(args),
                      getattr(args, 'output_dir', None), render_skin=not args.skip_skin)
