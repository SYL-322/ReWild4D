"""Artemis point-cloud geometry metrics; no checkpoint, renderer or model dependencies."""
import json
import os
import numpy as np
from scipy.spatial import cKDTree
from . import alignment as align

N_ALIGN_CANDIDATES = 5
F_THRESHOLDS = (0.01, 0.02, 0.05)

def source_frame_ids(source_path):
    """Original Artemis frame indices, in the same order and time convention as training."""
    with open(os.path.join(source_path, "artemis/transforms_artemis.json")) as f:
        frames = json.load(f)['frames']
    ids = [int(frame['time']) for frame in frames]
    if not ids or any(frame['time'] != index for frame, index in zip(frames, ids)):
        raise ValueError("Artemis source times must be nonempty integer frame indices")
    if ids[0] < 0 or any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError("Artemis source frame indices must be nonnegative and strictly increasing")
    return ids

def frame_metrics(pred, gt, diag):
    d_pred, _ = cKDTree(gt).query(pred)  # pred -> GT
    d_gt, _ = cKDTree(pred).query(gt)    # GT -> pred
    out = {'cd': float(0.5 * (d_pred.mean() + d_gt.mean())) / diag}
    for frac in F_THRESHOLDS:
        precision, recall = float((d_pred < frac * diag).mean()), float((d_gt < frac * diag).mean())
        out[f'f{round(frac * 100)}'] = 2 * precision * recall / (precision + recall + 1e-8)
    return out

def sequence_alignment(pred, gt):
    """similarity transform of the candidate frame with the lowest chamfer distance after ICP"""
    n = len(pred)
    candidates = sorted(set(np.linspace(0, n - 1, num=min(N_ALIGN_CANDIDATES, n)).astype(int).tolist()))
    best_M, best_residual = None, np.inf
    for f in candidates:
        M, history = align.similarity_icp(pred[f], gt[f], align.coarse_init(pred[f], gt[f]), max_iter=60)
        if history[-1] < best_residual:
            best_M, best_residual = M, history[-1]
    return best_M

def eval_points(pred, gt, diag):
    """mean over the frames of CD / diag and F@1/2/5%; pred and gt are compared frame by frame"""
    if len(pred) != len(gt) or len(pred) == 0:
        raise ValueError(f"Expected equal, nonempty matched frames; got prediction={len(pred)}, GT={len(gt)}")
    n = len(pred)
    M = sequence_alignment(pred, gt)
    per_frame = []
    for i in range(n):
        M_i, _ = align.similarity_icp(pred[i], gt[i], M, max_iter=30)
        per_frame.append(frame_metrics(align.apply(M_i, pred[i]), gt[i], diag))
    results = {k: float(np.mean([m[k] for m in per_frame])) for k in per_frame[0]}
    results['n_frames'] = n
    return results

def gt_for_scene(chamfer_root, source_path):
    """(gt file, rest-pose bbox diagonal) of <animal>/<motion>, or (None, None) if the geometry folder has no GT for it"""
    seq_dir = os.path.dirname(os.path.normpath(source_path))
    animal, motion = os.path.basename(os.path.dirname(seq_dir)), os.path.basename(seq_dir)
    gt_file = os.path.join(chamfer_root, "gt", f"{animal}_{motion}.npz")
    if not os.path.isfile(gt_file):
        return None, None
    meta = json.load(open(os.path.join(chamfer_root, "metadata.json"), 'r'))
    return gt_file, meta[animal]['rest_pose_bbox_diag']


def eval_prediction_file(prediction_file, gt_file, diag):
    """Compare exported world-space points using their original Artemis source-frame indices."""
    with np.load(prediction_file, allow_pickle=False) as archive:
        pred = archive['points']
        ids = archive['frame_ids']
    if pred.ndim != 3 or pred.shape[-1] != 3 or pred.shape[1] == 0 or not np.isfinite(pred).all():
        raise ValueError('Prediction points must be finite [frames, points, 3] arrays')
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) != len(pred) or len(ids) == 0:
        raise ValueError('frame_ids must contain one original integer source index per prediction frame')
    if ids[0] < 0 or np.any(np.diff(ids) <= 0):
        raise ValueError('frame_ids must be nonnegative and strictly increasing')
    if not np.isfinite(diag) or diag <= 0:
        raise ValueError('The rest-pose diagonal must be finite and positive')
    with np.load(gt_file, allow_pickle=False) as archive:
        gt = archive['points'].astype(np.float32)
    if ids[-1] >= len(gt):
        raise ValueError(f'Source frame {ids[-1]} is outside GT with {len(gt)} frames: {gt_file}')
    return eval_points(pred, gt[ids], diag)
