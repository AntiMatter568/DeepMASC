import mrcfile
import numpy as np
from scipy.ndimage import distance_transform_edt

INI_THRESHOLD = 0.01
RELION_MIN_R2_START = 9999.0  # RELION's initial "smallest squared distance" (src/mask.cpp:466)


def touches_box(mask):
    """True when any voxel on any of the six box faces is non-zero."""
    m = np.asarray(mask)
    faces = (m[0], m[-1], m[:, 0], m[:, -1], m[:, :, 0], m[:, :, -1])
    return bool(any(np.any(f != 0) for f in faces))


def relion_soft_edge(data, width, ini_threshold=INI_THRESHOLD, extend=0.0):
    """The soft mask relion_mask_create --ini_threshold 0.01 --extend_inimask e --width_soft_edge w makes.

    RELION 5.0.1 autoMask (src/mask.cpp:303-319, 446-497): voxels >= ini_threshold become 1; each 0
    voxel takes the smallest squared distance (voxel units) to a 1 voxel within a cube of half-size
    ceil(w), starting from 9999, and gets 0.5 + 0.5 cos(pi sqrt(r2) / w) when r2 < w^2. That is the
    Euclidean distance transform capped at 9999: a voxel with no mask voxel in its cube keeps 9999,
    which only matters beyond w = sqrt(9999) = 99.995 px, where RELION gives every such voxel the same
    small value. The cost does not grow with w.

    Padding (step B, src/mask.cpp:321-372, positive extend only): a 0 voxel strictly closer than
    `extend` to a 1 voxel becomes 1, and the soft edge is measured from this padded mask. Squared
    distances between voxel centres are integers, so they are rounded before the comparison.
    """
    if extend < 0:
        raise ValueError(f"negative padding (RELION's mask shrinking) is not supported: {extend}")
    binary = np.asarray(data) >= ini_threshold
    if extend > 0 and binary.any():
        d0 = np.asarray(distance_transform_edt(~binary), dtype=float)
        binary = binary | (np.rint(d0 * d0) < extend * extend)
    if binary.any():
        dist = np.asarray(distance_transform_edt(~binary), dtype=float)
        r2 = np.minimum(dist * dist, RELION_MIN_R2_START)
    else:
        r2 = np.full(binary.shape, RELION_MIN_R2_START)
    soft = np.where(r2 < width * width, 0.5 + 0.5 * np.cos(np.pi * np.sqrt(r2) / width), 0.0)
    return np.where(binary, 1.0, soft)


def write_soft_mask(src, dst, width, extend=0.0):
    """Write relion_soft_edge of the mask `src` to `dst` as float32, keeping the voxel size, origin,
    start indices and axis order of the input."""
    with mrcfile.open(src, permissive=True) as f:
        if f.header is None or f.data is None:
            raise ValueError(f"unreadable MRC: {src}")
        data = np.asarray(f.data)
        header = f.header.copy()
        voxel = f.voxel_size.copy()
    soft = relion_soft_edge(data, float(width), extend=float(extend)).astype(np.float32)
    with mrcfile.new(dst, overwrite=True) as g:
        g.set_data(soft)
        g.voxel_size = voxel
        out_header = g.header
        assert out_header is not None
        for key in ("origin", "nxstart", "nystart", "nzstart", "mapc", "mapr", "maps"):
            out_header[key] = header[key]
