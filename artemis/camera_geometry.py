"""Artemis alignment and virtual cameras, in column-vector world-to-camera convention."""
import os
import numpy as np


ARTEMIS_R_CAN = {
    "wolf": [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
    "bear": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    "duck": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
}
MIRROR = np.diag([-1., 1., 1., 1.])
CAMERA_REVISION = "analytic-fauna-bilateral-v1"


def artemis_animal(path):
    """<root>/<animal>/<motion>/cropped -> animal."""
    return os.path.normpath(path).split(os.sep)[-3]


def artemis_rotation(path):
    return np.asarray(ARTEMIS_R_CAN.get(artemis_animal(path), np.eye(3)), dtype=np.float32)


def alignment_from_transforms(contents, vertices):
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = float((high - low).max())
    scale = float(contents['c_max_dim']) / float(contents['scale_factor']) / extent
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Artemis alignment requires a nondegenerate prior and positive scale")
    return {'v_center': (low + high) / 2.,
            'c_center': np.asarray(contents['c_center'], dtype=np.float32),
            'scale': np.asarray(scale, dtype=np.float32)}


def bilateral_reflection(alignment, rotation):
    """Fauna's X=0 plane, rotated and translated by the training alignment."""
    rotation = np.asarray(rotation, dtype=np.float64)
    origin = np.asarray(alignment['c_center']) - float(alignment['scale']) * np.asarray(alignment['v_center'])
    reflection = np.eye(4)
    reflection[:3, :3] = rotation @ MIRROR[:3, :3] @ rotation.T
    reflection[:3, 3] = origin - reflection[:3, :3] @ origin
    return reflection


def mirrored_camera(camera, reflection):
    """Reflect the camera across the aligned bilateral plane.

    C_flip = M C S and P_flip = M P M give P_flip C_flip S = M P C.
    JSON matrices are transposed. The projection rule also supports off-center intrinsics.
    """
    view = MIRROR @ np.asarray(camera['viewmatrix'], dtype=np.float64).T @ reflection
    projection = MIRROR @ np.asarray(camera['projmatrix'], dtype=np.float64).T @ MIRROR
    return {**camera, 'viewmatrix': view.T.tolist(), 'projmatrix': projection.T.tolist(),
            'campos': np.linalg.inv(view)[:3, 3].tolist()}


def corrected_transforms(contents, alignment, rotation):
    reflection = bilateral_reflection(alignment, rotation)
    return {**contents, 'frames_flip': [mirrored_camera(frame, reflection) for frame in contents['frames']],
            'mirror_camera_rule': CAMERA_REVISION}
