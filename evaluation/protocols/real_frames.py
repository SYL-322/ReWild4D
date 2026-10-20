"""Match real-video predictions to frame identities before computing metrics."""
import json
from pathlib import Path


def load_robust_frames(data_path):
    with (Path(data_path) / 'fauna/transforms_wflip_wRobust.json').open() as stream:
        contents = json.load(stream)
    frames, flipped = contents['frames'], contents['frames_flip']
    if not frames or len(frames) != len(flipped):
        raise ValueError('Real-video annotations require nonempty, equally sized original and flip streams')
    names = [Path(frame['file_path']).stem for frame in frames]
    if len(names) != len(set(names)):
        raise ValueError('Real-video frame names must be unique')
    for original, flip in zip(frames, flipped):
        if original['file_path'] != flip['file_path'] or original['time'] != flip['time']:
            raise ValueError('Original and flip annotations must identify the same frame and timestamp')
    selected = [i for i, (original, flip) in enumerate(zip(frames, flipped))
                if original['is_robust'] or flip['is_robust']]
    return frames, selected


def _check_names(actual, expected, label):
    missing, extra = expected - actual, actual - expected
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} {label}; examples: {sorted(missing)[:5]}')
    if extra:
        raise ValueError(f'Unexpected {len(extra)} {label}; examples: {sorted(extra)[:5]}')


def matched_input_paths(eval_dir, source_path=None):
    root = Path(eval_dir)
    gt_dir, pred_dir = root / 'original_GT', root / 'original_pred'
    gt_names = {path.name[:-len('_gt_rgba.png')] for path in gt_dir.glob('*_gt_rgba.png')}
    pred_names = {path.name[:-len('_pred_rgba.png')] for path in pred_dir.glob('*_pred_rgba.png')}
    if source_path is None:
        expected = gt_names
    else:
        frames, selected = load_robust_frames(source_path)
        expected = {Path(frames[i]['file_path']).stem for i in selected if frames[i]['is_val']}
    if not expected:
        raise ValueError('Input-view evaluation requires nonempty held-out frame pairs')
    _check_names(gt_names, expected, 'input-view reference frames')
    _check_names(pred_names, expected, 'input-view prediction frames')
    names = sorted(expected)
    return ([str(pred_dir / f'{name}_pred_rgba.png') for name in names],
            [str(gt_dir / f'{name}_gt_rgba.png') for name in names])


def matched_novel_paths(novel_dir, frame_names, n_spin):
    root = Path(novel_dir)
    if not frame_names or len(frame_names) != len(set(frame_names)):
        raise ValueError('Novel-view evaluation requires nonempty, unique robust frames')
    expected = {f'{name}_spin_{view:02d}_pred_rgb.jpg' for name in frame_names for view in range(n_spin)}
    actual = {path.name for path in root.glob('*_spin_*_pred_rgb.jpg')}
    _check_names(actual, expected, 'novel-view prediction images')
    # Keep annotation order, so generated videos and reference videos share timestamps.
    return [[str(root / f'{name}_spin_{view:02d}_pred_rgb.jpg') for view in range(n_spin)]
            for name in frame_names]
