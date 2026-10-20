"""
Generate evaluation predictions from an optimized Artemis model.
Metric computation is in ../evaluation/.

novel views (every frame, from all Artemis cameras except the one of the training image, white background):
    <model_path>/eval_<iter>/<frame>/<view>.jpg        rendered frames
geometry (sampled Gaussian centers, mapped into the Artemis world):
    <model_path>/eval_<iter>/predicted_points.npz    points [frames, 10000, 3] and original frame_ids
The independent evaluator computes image and geometry metrics from these predictions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import prepare_models
from scene.dataset_readers import training_alignment
from utils.general_utils import set_random_seed
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_geometry import export_points


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def evaluation_metadata(args):
    """Record the rendering environment and arguments with the predictions."""
    import platform
    import PIL
    import piq
    from PIL import features
    return {
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'piq': piq.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(),
        'image_decoder': {'pillow': PIL.__version__, 'libjpeg': features.version('jpg'),
                          'libjpeg_turbo': features.version('libjpeg_turbo')},
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'arguments': vars(args),
    }


class ImageWriter:
    """saves images in background threads, so that JPEG / PNG encoding overlaps with rendering"""
    def __init__(self, workers=8):
        self.pool = ThreadPoolExecutor(workers)
        self.futures = []

    def save(self, image, path):
        """image: (C, H, W) float tensor in [0, 1], C = 3 (RGB) or 4 (RGBA)"""
        array = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        self.futures.append(self.pool.submit(lambda: Image.fromarray(array).save(path)))

    def close(self):
        """wait until every image is written (and raise the first error, if any)"""
        for future in self.futures:
            future.result()
        self.pool.shutdown()


class ValView:
    """an Artemis camera of artemis/val_info.json, with what the renderer needs"""
    def __init__(self, camera, alignment, size=512):
        # FoV rounded to float32, as stored in the camera tensors
        self.FoVx, self.FoVy = float(np.float32(camera['FoVx'])), float(np.float32(camera['FoVy']))
        self.image_height = self.image_width = size
        self.aux = {k: torch.tensor(camera[k], dtype=torch.float32).cuda() for k in ['viewmatrix', 'projmatrix', 'campos']}
        self.aux.update(alignment)


def load_val_info(source_path):
    """Load held-out views from val_info.json and the canonical-to-world transform from the input cameras."""
    val_info = json.load(open(os.path.join(source_path, "artemis/val_info.json"), 'r'))
    alignment = {k: torch.tensor(v, dtype=torch.float32).cuda() for k, v in training_alignment(source_path).items()}
    return val_info, alignment


@torch.no_grad()
def deform_frames(gaussians, deform_model, frame_ids):
    """(frame index, deformation stages) of every frame, in the original-image series"""
    signal = torch.tensor([1.0, 1.0], dtype=torch.float32).cuda()
    num_frames = len(frame_ids)
    for t, frame_id in enumerate(frame_ids):
        # Use source frame IDs, including gaps, to match optimization timestamps.
        time_input = torch.tensor([[(int(frame_id) + 1) / num_frames]], dtype=torch.float32).cuda()
        yield t, deform_model.train_step(gaussians, time_input, signal=signal)[0]


@torch.no_grad()
def render_val_views(dataset, gaussians, deform_model, pipeline, eval_dir):
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    val_info, alignment = load_val_info(dataset.source_path)
    frames = list(val_info['val_names'].items())

    writer = ImageWriter()
    for t, stages in tqdm(deform_frames(gaussians, deform_model, [frame for frame, _ in frames]), total=len(frames), desc="Rendering novel views"):
        frame, names = frames[t]
        os.makedirs(os.path.join(eval_dir, frame), exist_ok=True)
        for name in names:
            view = ValView(val_info['camera'][str(int(name.split('.')[0]))], alignment)
            results = render(view, stages[-1], gaussians.active_sh_degree, gaussians.max_sh_degree, pipeline, background)
            writer.save(torch.clamp(results["render"], 0.0, 1.0), os.path.join(eval_dir, frame, name))
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Artemis evaluation images and world-space point clouds")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--skip_views", action="store_true", help="skip generating novel-view predictions")
    parser.add_argument("--skip_chamfer", action="store_true", help="skip exporting geometry prediction points")
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    it = args.iteration
    print("Generating evaluation predictions for " + dataset.model_path)

    set_random_seed(seed=0)
    gaussians, deform_model = prepare_models(dataset, it)

    eval_dir = os.path.join(dataset.model_path, f"eval_{it}")
    os.makedirs(eval_dir, exist_ok=True)
    save_json(os.path.join(eval_dir, "prediction_metadata.json"), evaluation_metadata(args))
    save_json(os.path.join(eval_dir, "prediction_manifest.json"), {
        "schema_version": 1, "kind": "artemis", "source_path": dataset.source_path,
        "iteration": it, "geometry_prediction": "predicted_points.npz",
        "point_coordinate_system": "Artemis world",
    })
    if not getattr(args, "skip_views", False):
        render_val_views(dataset, gaussians, deform_model, pipe, eval_dir)

    if not getattr(args, "skip_chamfer", False):
        export_points(dataset.source_path, gaussians, deform_model, os.path.join(eval_dir, "predicted_points.npz"))
