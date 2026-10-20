"""Model-specific Artemis world-space point sampling, exported for independent evaluation."""
import numpy as np
import torch
from evaluation.protocols.constants import OPACITY_THRESHOLD, SCALE_THRESHOLD, N_POINTS, SEED
from evaluation.metrics.geometry import source_frame_ids
from scene.dataset_readers import training_alignment

@torch.no_grad()
def predicted_points(source_path, gaussians, deform_model):
    """(n_frames, N_POINTS, 3) float32 Gaussian centers in the Artemis world"""
    alignment = training_alignment(source_path)
    v_center, c_center, scale = [torch.tensor(alignment[k], dtype=torch.float32).cuda() for k in ['v_center', 'c_center', 'scale']]
    frame_ids = source_frame_ids(source_path)
    n_frame = len(frame_ids)
    signal = torch.tensor([1.0, 1.0], dtype=torch.float32).cuda()
    rng = np.random.default_rng(SEED)

    points = np.zeros((n_frame, N_POINTS, 3), dtype=np.float32)
    for i, frame_id in enumerate(frame_ids):
        time_input = torch.tensor([[(frame_id + 1) / n_frame]], dtype=torch.float32).cuda()
        deformed = deform_model.train_step(gaussians, time_input, signal=signal)[0][-1]
        xyz = (deformed.xyz - v_center) * scale + c_center
        keep = (deformed.opacity.squeeze(-1) > OPACITY_THRESHOLD) & (deformed.scaling.max(dim=1).values >= SCALE_THRESHOLD)
        kept = xyz[keep].cpu().numpy().astype(np.float32)
        if len(kept) == 0:
            kept = xyz.cpu().numpy().astype(np.float32)
        points[i] = kept[rng.choice(len(kept), N_POINTS, replace=len(kept) < N_POINTS)]
    return points


def export_points(source_path, gaussians, deform_model, output_file):
    points = predicted_points(source_path, gaussians, deform_model)
    frame_ids = np.asarray(source_frame_ids(source_path), dtype=np.int64)
    np.savez(output_file, points=points, frame_ids=frame_ids)
