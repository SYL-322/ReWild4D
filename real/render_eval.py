"""
Generate evaluation images from an optimized real-video model.
Metric computation is in ../evaluation/.

input views (validation frames, black background):
    <model_path>/eval_<iter>/original_{GT,pred}/       rendered frames
novel views (robust frames, 64 views around the animal, white background):
    <model_path>/eval_<iter>/novel_views_64/           rendered frames
The independent evaluator computes metrics from these images.
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
from gaussian_renderer import render, filter_gaussians_by_scale
from scene import Scene, prepare_models
from scene.gaussian_model import GaussianParams
from utils.general_utils import set_random_seed
from utils.visual_utils import rotate_gaussian, shift_by_campos
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.protocols.constants import N_SPIN


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


def deform_to_frame(gaussians, deform_model, view):
    fid = view.fid.cuda()
    pose = view.aux['pose'].cuda()
    gaussians_stages, _ = deform_model.train_step(gaussians, pose, fid.unsqueeze(0), signal=view.aux['signal'])
    return gaussians_stages[-1]


@torch.no_grad()
def render_input_views(dataset, gaussians, deform_model, iteration, pipeline, eval_dir):
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    scene = Scene(dataset, gaussians, load_iteration=iteration, load_flip=False, load_all=False)
    gt_dir = os.path.join(eval_dir, "original_GT")
    pred_dir = os.path.join(eval_dir, "original_pred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    writer = ImageWriter()
    for view_dict in tqdm(scene.getTestCameras(), desc="Rendering input views"):
        view = view_dict["original"]
        view.load2device()
        gt_img = view.original_image.cuda()
        gt_mask = view.original_mask.cuda()
        gt_img[:, gt_mask.squeeze() == 0] = background[:, None]

        results = render(view, deform_to_frame(gaussians, deform_model, view), gaussians.active_sh_degree, gaussians.max_sh_degree,
                         pipeline, background, use_filter=True)
        rgba = torch.cat([torch.clamp(results["render"], 0.0, 1.0), torch.clamp(results["alpha"], 0.0, 1.0)], dim=0)
        writer.save(rgba, os.path.join(pred_dir, f"{view.image_name}_pred_rgba.png"))
        writer.save(torch.cat([gt_img, gt_mask], dim=0), os.path.join(gt_dir, f"{view.image_name}_gt_rgba.png"))
    writer.close()


@torch.no_grad()
def render_novel_views(dataset, gaussians, deform_model, iteration, pipeline, novel_view_dir):
    """N_SPIN views per frame, rotating the animal around the vertical axis"""
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    scene = Scene(dataset, gaussians, load_iteration=iteration, load_flip=False, load_all=True)
    os.makedirs(novel_view_dir, exist_ok=True)

    writer = ImageWriter()
    for view_dict in tqdm(scene.getTestCameras(), desc="Rendering novel views"):
        view = view_dict["original"]
        view.load2device()
        centered = shift_by_campos(deform_to_frame(gaussians, deform_model, view), view)
        # Scale filtering is rotation-invariant. Keep the unfiltered center as the spin pivot.
        center = centered.xyz.mean(0)
        filtered = GaussianParams(*filter_gaussians_by_scale(*centered))
        for vid in range(N_SPIN):
            spin_tuple = rotate_gaussian(filtered, 2 * np.pi * vid / N_SPIN, center=center)
            results = render(view, spin_tuple, gaussians.active_sh_degree, gaussians.max_sh_degree, pipeline, background)
            writer.save(torch.clamp(results["render"], 0.0, 1.0), os.path.join(novel_view_dir, f"{view.image_name}_spin_{vid:02d}_pred_rgb.jpg"))
    writer.close()


def save_gifs(novel_view_dir, save_dir):
    """visualization only: 4D (view rotates with time), 3D (spin at 10 frames) and fixed-view videos"""
    n = N_SPIN
    names = sorted(f.split('_spin')[0] for f in os.listdir(novel_view_dir) if f.endswith("_spin_00_pred_rgb.jpg"))
    os.makedirs(save_dir, exist_ok=True)

    def save(paths, name):
        frames = [Image.open(p) for p in paths]
        frames[0].save(os.path.join(save_dir, name), save_all=True, append_images=frames[1:], loop=0, duration=100)
    path = lambda name, vid: os.path.join(novel_view_dir, f"{name}_spin_{vid:02d}_pred_rgb.jpg")

    loop = (names + names[::-1]) * int(np.ceil(n / (2 * len(names))))  # ping-pong padding to at least n frames
    loop = loop[:max(n, len(names))]
    save([path(name, t % n) for t, name in enumerate(loop)], f"4d_spin_NV{n}.gif")
    for t in np.linspace(0, len(names) - 1, min(10, len(names))).astype(int):
        save([path(names[t], vid) for vid in range(n)], f"3d_spin_{names[t]}_NV{n}.gif")
    for vid in [0, n // 4, n // 2, 3 * n // 4]:
        save([path(name, vid) for name in names], f"2d_view{vid:02d}_NV{n}.gif")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate real-video evaluation predictions")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--skip_input", action="store_true", help="skip generating input-view predictions")
    parser.add_argument("--skip_novel", action="store_true", help="skip generating novel-view predictions")
    parser.add_argument("--save_gif", action="store_true", help="also save novel-view gifs")
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
        "schema_version": 1, "kind": "real", "source_path": dataset.source_path,
        "iteration": it, "n_spin": N_SPIN,
    })

    if not args.skip_input:
        render_input_views(dataset, gaussians, deform_model, it, pipe, eval_dir)

    if not args.skip_novel:
        novel_view_dir = os.path.join(eval_dir, f"novel_views_{N_SPIN}")
        render_novel_views(dataset, gaussians, deform_model, it, pipe, novel_view_dir)
        if args.save_gif:
            save_gifs(novel_view_dir, os.path.join(eval_dir, "gifs"))
