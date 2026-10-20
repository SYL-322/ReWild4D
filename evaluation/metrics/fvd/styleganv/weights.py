"""Fetch and verify the I3D checkpoint used for FVD."""
import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen


I3D_URL = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
I3D_SHA256 = 'bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2'


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_i3d_weights():
    """Validate cached weights or download them without exposing partial files.

    Use REWILD_I3D_PATH if set; otherwise cache the checkpoint beside this module.
    """
    override = os.environ.get('REWILD_I3D_PATH')
    path = Path(override).expanduser() if override else Path(__file__).with_name('i3d_torchscript.pt')
    if path.is_file() and _sha256(path) == I3D_SHA256:
        return path
    instructions = (
        f'Download {I3D_URL} manually and set REWILD_I3D_PATH to the downloaded file. '
        f'Expected SHA-256: {I3D_SHA256}.'
    )
    if override:
        raise RuntimeError(f'I3D checkpoint is missing or has an invalid SHA-256: {path}. {instructions}')

    print(f'Downloading verified I3D weights to {path}', flush=True)
    for attempt in range(3):
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Unique files in the cache directory also allow concurrent evaluators.
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                request = Request(I3D_URL, headers={'User-Agent': 'ReWild4D'})
                with urlopen(request, timeout=30) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        output.write(block)
            actual = _sha256(temporary)
            if actual != I3D_SHA256:
                raise ValueError(f'I3D SHA-256 mismatch: got {actual}, expected {I3D_SHA256}')
            # The cache filename only ever receives a complete, verified checkpoint.
            os.replace(temporary, path)
            return path
        except (OSError, HTTPException, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f'Unable to download verified I3D weights to {path} after 3 attempts. {instructions}') from error
            print(f'I3D download attempt {attempt + 1}/3 failed: {error}; retrying.', flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        time.sleep(attempt + 1)

