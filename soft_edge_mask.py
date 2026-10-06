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


def write_soft_mask(src, dst, width, extend=0.0, ini_threshold=INI_THRESHOLD):
    """Write relion_soft_edge of the mask `src` to `dst` as float32, keeping the voxel size, origin,
    start indices and axis order of the input. Returns the soft mask array."""
    with mrcfile.open(src, permissive=True) as f:
        if f.header is None or f.data is None:
            raise ValueError(f"unreadable MRC: {src}")
        data = np.asarray(f.data)
        header = f.header.copy()
        voxel = f.voxel_size.copy()
    soft = relion_soft_edge(data, float(width), ini_threshold=float(ini_threshold),
                            extend=float(extend)).astype(np.float32)
    with mrcfile.new(dst, overwrite=True) as g:
        g.set_data(soft)
        g.voxel_size = voxel
        out_header = g.header
        assert out_header is not None
        for key in ("origin", "nxstart", "nystart", "nzstart", "mapc", "mapr", "maps"):
            out_header[key] = header[key]
    return soft


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
    soft = write_soft_mask(args.input, args.output, width_px, extend=pad_px, ini_threshold=args.ini_threshold)
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
