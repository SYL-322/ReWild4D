"""Shared utilities from rewild_common.general_utils."""
from pathlib import Path as _Path
import sys as _sys

_repository_root = str(_Path(__file__).resolve().parents[2])
if _repository_root not in _sys.path:
    _sys.path.insert(0, _repository_root)

from rewild_common.general_utils import *
