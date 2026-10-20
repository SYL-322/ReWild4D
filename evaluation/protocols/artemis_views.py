"""Validate the exact held-out Artemis image cohort before computing metrics."""
from pathlib import Path


def matched_view_paths(eval_dir, gt_dir, val_names):
    eval_dir, gt_dir = Path(eval_dir), Path(gt_dir)
    expected = {(str(frame), str(name)) for frame, names in val_names.items() for name in names}
    if not expected or any(not names or len(names) != len(set(names)) for names in val_names.values()):
        raise ValueError('Artemis val_names must contain nonempty, unique held-out views')
    actual = {(frame.name, image.name) for frame in eval_dir.iterdir() if frame.is_dir() and frame.name.isdecimal()
              for image in frame.iterdir() if image.is_file()}
    missing, extra = expected - actual, actual - expected
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} Artemis prediction images; examples: {sorted(missing)[:5]}')
    if extra:
        raise ValueError(f'Unexpected {len(extra)} Artemis prediction images outside val_names; examples: {sorted(extra)[:5]}')
    pairs = [(eval_dir / frame / name, gt_dir / 'images' / frame / name) for frame, name in sorted(expected)]
    missing_gt = [str(gt) for _, gt in pairs if not gt.is_file()]
    if missing_gt:
        raise FileNotFoundError(f'Missing {len(missing_gt)} Artemis GT images; examples: {missing_gt[:5]}')
    return [str(pred) for pred, _ in pairs], [str(gt) for _, gt in pairs]
