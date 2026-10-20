"""Metric environment metadata, independent of model/checkpoint modules."""
import platform
from importlib.metadata import version, PackageNotFoundError


def metric_environment():
    result = {'python': platform.python_version()}
    import numpy as np
    result['numpy'] = np.__version__
    for package in ['scipy', 'piq', 'lpips']:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    try:
        import PIL
        from PIL import features
        result['image_decoder'] = {'pillow': PIL.__version__, 'libjpeg': features.version('jpg'),
                                   'libjpeg_turbo': features.version('libjpeg_turbo')}
    except ImportError:
        pass
    try:
        import torch
        result.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                      cudnn_deterministic=torch.backends.cudnn.deterministic,
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                      matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    except ImportError:
        pass
    return result
