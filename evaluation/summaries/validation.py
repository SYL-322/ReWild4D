"""Read complete, finite per-scene metrics before adding them to a summary."""
import json
import math


IMAGE_METRICS = ('psnr', 'ssim', 'lpips_alex')
GEOMETRY_METRICS = ('cd', 'f1', 'f2', 'f5', 'n_frames')


def read_metrics(path, required):
    with open(path) as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a JSON object')
    missing = set(required) - data.keys()
    if missing:
        raise ValueError(f'{path}: missing metrics {sorted(missing)}')
    for key in required:
        value = data[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{path}: {key} must be a finite number, got {value!r}')
        if key == 'n_frames' and (type(value) is not int or value <= 0):
            raise ValueError(f'{path}: n_frames must be a positive integer')
    return {key: data[key] for key in required}
