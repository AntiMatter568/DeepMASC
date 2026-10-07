"""Hand-built masks and the pre-reuse soft edge, shared by the soft edge tests (not a test module)."""
import numpy as np
from scipy.ndimage import distance_transform_edt

WIDTHS = [1.5, 5, 10, 20, 60, 100, 150]
EXTENDS = [0.0, 3.5]


def legacy_soft_edge(data, width, ini_threshold=0.01, extend=0.0):
    """relion_soft_edge as it was before the distance was shared between widths (frozen copy)."""
    binary = np.asarray(data) >= ini_threshold
    if extend > 0 and binary.any():
        d0 = np.asarray(distance_transform_edt(~binary), dtype=float)
        binary = binary | (np.rint(d0 * d0) < extend * extend)
    if binary.any():
        dist = np.asarray(distance_transform_edt(~binary), dtype=float)
        r2 = np.minimum(dist * dist, 9999.0)
    else:
        r2 = np.full(binary.shape, 9999.0)
    soft = np.where(r2 < width * width, 0.5 + 0.5 * np.cos(np.pi * np.sqrt(r2) / width), 0.0)
    return np.where(binary, 1.0, soft)


def shapes():
    out = {}
    z, y, x = np.indices((40, 44, 36))
    out["sphere"] = ((z - 20) ** 2 + (y - 22) ** 2 + (x - 18) ** 2 <= 49).astype(np.float32)
    out["two_blobs"] = np.zeros((40, 44, 36), np.float32)
    out["two_blobs"][5:9, 5:9, 5:9] = 1
    out["two_blobs"][30:36, 30:40, 20:30] = 1
    out["empty"] = np.zeros((20, 22, 18), np.float32)
    out["full"] = np.ones((12, 12, 12), np.float32)
    out["touches_faces"] = np.zeros((30, 30, 30), np.float32)
    out["touches_faces"][0:5, 10:20, 10:20] = 1
    out["corner_voxel"] = np.zeros((30, 31, 32), np.float32)
    out["corner_voxel"][1, 1, 1] = 1
    rng = np.random.default_rng(3)
    out["sparse_soft"] = (rng.random((30, 28, 26)) * (rng.random((30, 28, 26)) < 0.03)).astype(np.float32)
    return out
