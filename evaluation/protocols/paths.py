"""Resolve prediction and ground-truth paths."""
import os
from pathlib import Path


def real_novel_path(model_path, iteration, name):
    """Read unified outputs, falling back to the older split directory layout."""
    current = Path(model_path) / f'eval_{iteration}' / name
    previous = Path(model_path) / f'eval_{iteration}_filter' / name
    return current if current.exists() or not previous.exists() else previous


def data_root(source_path):
    """Shared data root for nested train/ and flat sequence layouts."""
    seq_dir = os.path.dirname(os.path.normpath(source_path))
    parent = os.path.dirname(os.path.dirname(seq_dir))
    if os.path.basename(parent) == "train":
        return os.path.dirname(os.path.dirname(parent))
    return os.path.dirname(parent)

def default_gt_path(source_path):
    """Multi-view GT for <data>/<dataset>/train/<animal>/<motion>/cropped."""
    seq_dir = os.path.dirname(os.path.normpath(source_path))
    parent = os.path.dirname(os.path.dirname(seq_dir))
    animal, motion = os.path.basename(os.path.dirname(seq_dir)), os.path.basename(seq_dir)
    if os.path.basename(parent) == "train":
        return os.path.join(os.path.dirname(parent), "eval", "views", animal, motion)
    return os.path.join(data_root(source_path), "Artemis", animal, motion)

def default_chamfer_root(source_path):
    """Geometry GT under eval/geometry, or chamfer_eval for flat sequence layouts."""
    seq_dir = os.path.dirname(os.path.normpath(source_path))
    parent = os.path.dirname(os.path.dirname(seq_dir))
    if os.path.basename(parent) == "train":
        return os.path.join(os.path.dirname(parent), "eval", "geometry")
    return os.path.join(data_root(source_path), "chamfer_eval")
