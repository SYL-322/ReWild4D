"""
Similarity alignment (rotation, translation, uniform scale) of a predicted point cloud to a ground-truth point cloud,
without correspondences: an initial guess from the 24 axis-aligned rotations, refined by similarity-ICP.
A transform is a 4 x 4 matrix M; a point x maps to M[:3, :3] @ x + M[:3, 3].
"""
import numpy as np
from scipy.spatial import cKDTree


def apply(M, points):
    return points @ M[:3, :3].T + M[:3, 3]


def chamfer(a, b):
    """symmetric chamfer distance, 0.5 * (mean a -> b + mean b -> a)"""
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return 0.5 * (da.mean() + db.mean())


def umeyama(src, dst, with_scale=True):
    """closed-form similarity transform mapping paired points src -> dst"""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((dst_c.T @ src_c) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / max((src_c ** 2).sum() / n, 1e-12) if with_scale else 1.0
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = mu_d - s * R @ mu_s
    return M


def axis_rotations():
    """the 24 rotations of a cube"""
    rotations = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    R = np.zeros((3, 3))
                    R[0], R[1], R[2] = sx * np.eye(3)[perm[0]], sy * np.eye(3)[perm[1]], sz * np.eye(3)[perm[2]]
                    if abs(np.linalg.det(R) - 1.0) < 1e-6:
                        rotations.append(R)
    return rotations


def coarse_init(pred, gt):
    """centroids and spreads matched, best of the 24 axis-aligned rotations"""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    s = gt.std(0).mean() / max(pred.std(0).mean(), 1e-12)
    best_cd, best_M = np.inf, None
    for R in axis_rotations():
        M = np.eye(4)
        M[:3, :3] = s * R
        M[:3, 3] = mu_g - s * R @ mu_p
        cd = chamfer(apply(M, pred), gt)
        if cd < best_cd:
            best_cd, best_M = cd, M
    return best_M


def similarity_icp(pred, gt, M_init, max_iter, tol=1e-7, with_scale=True):
    """
    ICP with nearest neighbours in both directions (pred -> GT and GT -> pred) as correspondences.
    Returns the transform and the chamfer distance after each iteration.
    """
    M, history, prev_cd = M_init.copy(), [], np.inf
    for _ in range(max_iter):
        cur = apply(M, pred)
        _, idx_p2g = cKDTree(gt).query(cur)
        _, idx_g2p = cKDTree(cur).query(gt)
        M = umeyama(np.concatenate([pred, pred[idx_g2p]], axis=0), np.concatenate([gt[idx_p2g], gt], axis=0), with_scale)
        cd = chamfer(apply(M, pred), gt)
        history.append(cd)
        if abs(prev_cd - cd) < tol:
            break
        prev_cd = cd
    return M, history
