import argparse
import sys

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


DEFAULT_EDT_BACKEND = "scipy"
DEFAULT_TAICHI_ARCH = "auto"
EDT_BACKENDS = ("scipy", "taichi", "auto")
TAICHI_ARCHS = ("cpu", "cuda", "auto")


class EdtBackendError(RuntimeError):
    """An explicitly requested distance backend could not be used."""


def _scipy_squared_distance(binary):
    dist = np.asarray(distance_transform_edt(~binary), dtype=float)
    return dist * dist


def _taichi_squared_distance(binary, arch, threads):
    """Squared distance to the nearest True voxel from the optional Taichi module; EdtBackendError if
    Taichi is missing or does not initialise."""
    try:
        import soft_edge_taichi
    except ImportError as e:
        raise EdtBackendError(f"the Taichi backend needs the optional 'taichi' package (1.7.x): {e}") from e
    try:
        soft_edge_taichi.init(arch, threads)
    except Exception as e:
        raise EdtBackendError(f"Taichi did not initialise (arch {arch}): {e}") from e
    return soft_edge_taichi.edt_sq(binary).astype(float)


def squared_distance_function(edt_backend=DEFAULT_EDT_BACKEND, taichi_arch=DEFAULT_TAICHI_ARCH, threads=None):
    """A function binary -> squared distance (float array, voxel units) of every voxel to the nearest True voxel.

    edt_backend "scipy" is scipy.ndimage.distance_transform_edt. "taichi" is the exact integer transform of
    soft_edge_taichi (CPU threads or CUDA, per taichi_arch) and raises EdtBackendError when Taichi is
    missing or does not start. "auto" is Taichi when it starts, else scipy after a printed note.
    Squared distances are the same integers on every backend.
    """
    if callable(edt_backend):
        return edt_backend
    if edt_backend not in EDT_BACKENDS:
        raise ValueError(f"edt_backend must be one of {EDT_BACKENDS}, not {edt_backend!r}")
    if taichi_arch not in TAICHI_ARCHS:
        raise ValueError(f"taichi_arch must be one of {TAICHI_ARCHS}, not {taichi_arch!r}")
    if edt_backend == "scipy":
        return _scipy_squared_distance
    probe = np.zeros((1, 1, 2), dtype=bool)
    probe[0, 0, 0] = True
    try:
        _taichi_squared_distance(probe, taichi_arch, threads)
    except EdtBackendError as e:
        if edt_backend == "taichi":
            raise
        print(f"Taichi distance backend not available ({e}); using scipy")
        return _scipy_squared_distance
    return lambda binary: _taichi_squared_distance(binary, taichi_arch, threads)


def capped_squared_distance(data, ini_threshold=INI_THRESHOLD, extend=0.0, edt_backend=DEFAULT_EDT_BACKEND,
                            taichi_arch=DEFAULT_TAICHI_ARCH, threads=None):
    """(binary mask, squared distance capped at 9999) from which the soft mask of any width follows.

    The binary mask is data >= ini_threshold, padded when extend > 0. The squared distance (voxel units) of
    every voxel to the nearest mask voxel is capped at RELION's 9999 and is 9999 everywhere for an empty
    mask. One transform for the mask, one more for the padding: the soft masks of all widths share them.

    Padding (step B, src/mask.cpp:321-372, positive extend only): a 0 voxel strictly closer than
    `extend` to a 1 voxel becomes 1, and the soft edge is measured from this padded mask. Squared
    distances between voxel centres are integers, so they are rounded before the comparison.
    """
    if extend < 0:
        raise ValueError(f"negative padding (RELION's mask shrinking) is not supported: {extend}")
    edt = squared_distance_function(edt_backend, taichi_arch, threads)
    binary = np.asarray(data) >= ini_threshold
    if extend > 0 and binary.any():
        d0 = edt(binary)
        binary = binary | (np.rint(d0) < extend * extend)
    if binary.any():
        r2 = np.minimum(edt(binary), RELION_MIN_R2_START)
    else:
        r2 = np.full(binary.shape, RELION_MIN_R2_START)
    return binary, r2


def soft_edge_from_distance(binary, r2, width):
    """The soft mask of one width from capped_squared_distance: 1 inside the mask, and
    0.5 + 0.5 cos(pi sqrt(r2) / width) where r2 < width^2, else 0.

    The cosine is evaluated only on the voxels inside the edge, which are the same values a full-array
    evaluation gives for them.
    """
    soft = np.zeros(r2.shape)
    edge = r2 < width * width
    soft[edge] = 0.5 + 0.5 * np.cos(np.pi * np.sqrt(r2[edge]) / width)
    soft[binary] = 1.0
    return soft


def relion_soft_edge(data, width, ini_threshold=INI_THRESHOLD, extend=0.0, edt_backend=DEFAULT_EDT_BACKEND,
                     taichi_arch=DEFAULT_TAICHI_ARCH, threads=None):
    """The soft mask relion_mask_create --ini_threshold 0.01 --extend_inimask e --width_soft_edge w makes.

    RELION 5.0.1 autoMask (src/mask.cpp:303-319, 446-497): voxels >= ini_threshold become 1; each 0
    voxel takes the smallest squared distance (voxel units) to a 1 voxel within a cube of half-size
    ceil(w), starting from 9999, and gets 0.5 + 0.5 cos(pi sqrt(r2) / w) when r2 < w^2. That is the
    Euclidean distance transform capped at 9999: a voxel with no mask voxel in its cube keeps 9999,
    which only matters beyond w = sqrt(9999) = 99.995 px, where RELION gives every such voxel the same
    small value. The cost does not grow with w.

    Padding (positive extend only) is described at capped_squared_distance. To make the soft masks of
    several widths from one mask, call capped_squared_distance once and soft_edge_from_distance per width.
    """
    binary, r2 = capped_squared_distance(data, ini_threshold, extend, edt_backend, taichi_arch, threads)
    return soft_edge_from_distance(binary, r2, width)


class SoftEdgeDistance:
    """The soft masks of several widths from one binary mask file, with one distance transform.

    The backend is chosen when the object is made (an unavailable explicit backend fails there). The mask is
    read and the distance computed at the first write(), once (twice with padding); every
    width after that only evaluates the cosine. write() keeps the voxel size, origin, start indices and
    axis order of the input, as write_soft_mask does.
    """

    def __init__(self, src, extend=0.0, ini_threshold=INI_THRESHOLD, edt_backend=DEFAULT_EDT_BACKEND,
                 taichi_arch=DEFAULT_TAICHI_ARCH, threads=None):
        self.src = src
        self.extend = float(extend)
        self.ini_threshold = float(ini_threshold)
        self.edt = squared_distance_function(edt_backend, taichi_arch, threads)  # fails early if unavailable
        self._state = None

    def _load(self):
        if self._state is None:
            with mrcfile.open(self.src, permissive=True) as f:
                if f.header is None or f.data is None:
                    raise ValueError(f"unreadable MRC: {self.src}")
                data = np.asarray(f.data)
                header = f.header.copy()
                voxel = f.voxel_size.copy()
            binary, r2 = capped_squared_distance(data, self.ini_threshold, self.extend, self.edt)
            self._state = (binary, r2, header, voxel)
        return self._state

    def write(self, dst, width):
        """Write the soft mask of `width` to `dst` as float32; returns the soft mask array."""
        binary, r2, header, voxel = self._load()
        soft = soft_edge_from_distance(binary, r2, float(width)).astype(np.float32)
        with mrcfile.new(dst, overwrite=True) as g:
            g.set_data(soft)
            g.voxel_size = voxel
            out_header = g.header
            assert out_header is not None
            for key in ("origin", "nxstart", "nystart", "nzstart", "mapc", "mapr", "maps"):
                out_header[key] = header[key]
        return soft


def write_soft_mask(src, dst, width, extend=0.0, ini_threshold=INI_THRESHOLD, edt_backend=DEFAULT_EDT_BACKEND,
                    taichi_arch=DEFAULT_TAICHI_ARCH, threads=None):
    """Write relion_soft_edge of the mask `src` to `dst` as float32, keeping the voxel size, origin,
    start indices and axis order of the input. Returns the soft mask array."""
    return SoftEdgeDistance(src, extend, ini_threshold, edt_backend, taichi_arch, threads).write(dst, float(width))


CRYOSPARC_V4 = "cryosparc_v4"
CRYOSPARC_V5 = "cryosparc_v5"


def cryosparc_soft_edge_px(rule, angpix, resolution):
    """(padding, edge) in pixels of the map for one cryoSPARC soft-edge rule.

    cryosparc_v4: 6 A padding + 6 A edge. cryosparc_v5: 2R padding + 3R edge, R the resolution in A.
    """
    if rule == CRYOSPARC_V4:
        pad_a, edge_a = 6.0, 6.0
    elif rule == CRYOSPARC_V5:
        if resolution is None or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("the cryoSPARC v5 soft edge needs the resolution R in A")
        pad_a, edge_a = 2.0 * resolution, 3.0 * resolution
    else:
        raise ValueError(f"unknown cryoSPARC rule {rule!r}")
    return pad_a / angpix, edge_a / angpix


def _num(x):
    return f"{x:.6g}"


def add_soft_edge_arguments(p):
    """Add the width, padding, threshold and preset options to the parser `p`."""
    p.add_argument("--width_px", type=float, help="soft edge width in pixels (RELION --width_soft_edge)")
    p.add_argument("--width_A", type=float, help="soft edge width in A (converted with the voxel size)")
    p.add_argument("--extend_px", type=float, help="padding in pixels (RELION --extend_inimask), default 0")
    p.add_argument("--extend_A", type=float, help="padding in A (converted with the voxel size)")
    p.add_argument("--ini_threshold", type=float, default=INI_THRESHOLD,
                   help="voxels at or above this value are the binary mask (default %(default)s)")
    p.add_argument("--preset", choices=[CRYOSPARC_V4, CRYOSPARC_V5],
                   help="cryosparc_v4: 6 A padding + 6 A edge; cryosparc_v5: 2R padding + 3R edge (needs "
                        "--resolution). Excludes explicit width and padding.")
    p.add_argument("--resolution", type=float, help="R in A, for --preset cryosparc_v5")
    add_edt_backend_arguments(p)


def add_edt_backend_arguments(p):
    """Add the distance-transform backend options to the parser `p`."""
    p.add_argument("--edt_backend", choices=EDT_BACKENDS, default=DEFAULT_EDT_BACKEND,
                   help="distance transform: scipy (default), taichi (optional package, faster on boxes of 512 "
                        "voxels or more with several CPU threads or a GPU; fails if unavailable), or auto "
                        "(taichi if it starts, else scipy)")
    p.add_argument("--taichi_arch", choices=TAICHI_ARCHS, default=DEFAULT_TAICHI_ARCH,
                   help="Taichi device: cpu, cuda, or auto (cuda if present, else cpu); default %(default)s")


def build_parser():
    p = argparse.ArgumentParser(
        description="Soft mask from a binary mask, equal to relion_mask_create --ini_threshold T "
                    "--extend_inimask E --width_soft_edge W. Width and padding are given in pixels or in A, "
                    "one unit per quantity, or by a cryoSPARC preset.")
    p.add_argument("-i", "--input", required=True, help="binary mask (MRC)")
    p.add_argument("-o", "--output", required=True, help="soft mask to write (MRC, float32)")
    add_soft_edge_arguments(p)
    return p


def voxel_size_of(parser, path):
    """Isotropic voxel size in A of the MRC `path`; exits through the parser if absent or anisotropic."""
    with mrcfile.open(path, permissive=True) as f:
        vx, vy, vz = (float(f.voxel_size.x), float(f.voxel_size.y), float(f.voxel_size.z))
    if not vx > 0:
        parser.error(f"the input has no voxel size: {path}")
    if not np.allclose([vy, vz], vx, rtol=1e-3):
        parser.error(f"anisotropic voxel size ({vx}, {vy}, {vz}) is not supported")
    return vx


def resolve_soft_edge(parser, args, angpix):
    """(width_px, padding_px, width_A, padding_A) from the parsed arguments; exits through the parser on
    conflicting or missing options."""
    explicit = [args.width_px, args.width_A, args.extend_px, args.extend_A]
    if args.preset:
        if any(v is not None for v in explicit):
            parser.error("--preset excludes --width_px/--width_A/--extend_px/--extend_A")
        if args.preset == CRYOSPARC_V5 and args.resolution is None:
            parser.error("--preset cryosparc_v5 needs --resolution R (in A)")
        if args.preset == CRYOSPARC_V4 and args.resolution is not None:
            parser.error("--resolution is only used by --preset cryosparc_v5")
        pad_px, width_px = cryosparc_soft_edge_px(args.preset, angpix, args.resolution)
        return width_px, pad_px, width_px * angpix, pad_px * angpix
    if args.resolution is not None:
        parser.error("--resolution is only used by --preset cryosparc_v5")
    if args.width_px is not None and args.width_A is not None:
        parser.error("give the width in pixels or in A, not both")
    if args.extend_px is not None and args.extend_A is not None:
        parser.error("give the padding in pixels or in A, not both")
    if args.width_px is None and args.width_A is None:
        parser.error("a width is required (--width_px or --width_A), or a --preset")
    width_px = args.width_px if args.width_px is not None else args.width_A / angpix
    if args.extend_px is not None:
        pad_px = args.extend_px
    elif args.extend_A is not None:
        pad_px = args.extend_A / angpix
    else:
        pad_px = 0.0
    if not width_px > 0:
        parser.error("the width must be positive")
    if pad_px < 0:
        parser.error("the padding must not be negative")
    return width_px, pad_px, width_px * angpix, pad_px * angpix


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    vx = voxel_size_of(parser, args.input)
    width_px, pad_px, width_a, pad_a = resolve_soft_edge(parser, args, vx)
    try:
        soft = write_soft_mask(args.input, args.output, width_px, extend=pad_px, ini_threshold=args.ini_threshold,
                               edt_backend=args.edt_backend, taichi_arch=args.taichi_arch)
    except EdtBackendError as e:
        parser.error(str(e))
    touches = touches_box(soft)
    print(f"soft edge: width {_num(width_px)} px ({_num(width_a)} A), padding {_num(pad_px)} px ({_num(pad_a)} A), "
          f"ini_threshold {_num(args.ini_threshold)}, voxel size {_num(vx)} A")
    print(f"touches box face: {'yes' if touches else 'no'}")
    if touches:
        print("WARNING: the soft mask is non-zero on a box face; the soft edge is clipped by the box.")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
