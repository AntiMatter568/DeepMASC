"""Tests for gmm_mask's resize path: maps with 5e6 or more non-zero voxels inside the spherical
pre-mask are fitted on a downsampled copy and predicted on the full map, so one density value must
have one feature value in both arrays. These tests run gmm_mask on full-size synthetic maps and take
a few minutes each."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from skimage.morphology import ball, closing, opening

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import contour  # noqa: E402

# the commit before the resize-path fix; the no-resize path must give its output unchanged
BEFORE_FIX = "798b0b9"


def _flat_solvent_map(path, n=232, seed=0):
    """Sharp-edged protein balls of density 1 in solvent flattened to [-0.002, 0.002]: the
    solvent is one 8-bit level wide, so its gradient is 0 and its VBGMM component is a narrow
    spike at density 0. The 95% sphere of a 232^3 box holds more than 5e6 voxels."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(-0.002, 0.002, (n, n, n)).astype(np.float32)
    z, y, x = np.ogrid[:n, :n, :n]
    c = (n - 1) / 2
    for _ in range(12):
        p = c + rng.uniform(-60, 60, 3)
        r = rng.uniform(12, 22)
        m[(z - p[0]) ** 2 + (y - p[1]) ** 2 + (x - p[2]) ** 2 <= r * r] = 1.0
    with mrcfile.new(path, overwrite=True) as f:
        f.set_data(m)
        f.voxel_size = 1.0
    return m


def _run(module, map_path, out):
    return module.gmm_mask(input_map_path=str(map_path), output_folder=str(out), num_components=2,
                           use_grad=True, n_init=3, plot_all=False, morph_radius=3, mask_diameter=95,
                           aggressive=False)


def _mask(path):
    with mrcfile.open(path) as f:
        return f.data > 0.5  # type: ignore[operator]


def test_flat_solvent_large_map_does_not_collapse(tmp_path):
    m = _flat_solvent_map(tmp_path / "flat.mrc")
    in_sphere = np.where(contour.create_spherical_mask(m.shape, radius=95), m, 0) != 0
    assert in_sphere.sum() >= 5e6  # the resize path is taken
    _run(contour, tmp_path / "flat.mrc", tmp_path / "out")
    cons = _mask(tmp_path / "out" / "prot_mask.mrc")
    # a collapse returns the all-protein mask: every non-zero voxel in the sphere, closed and opened
    all_protein = opening(closing(in_sphere, ball(3)), ball(3))
    assert not np.array_equal(cons, all_protein)
    assert cons[m == 1.0].all()  # the protein balls are kept
    assert cons.sum() < 2 * (m == 1.0).sum()  # and the solvent is not


def _before_fix_module(tmp_path):
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BEFORE_FIX}:contour.py"],
                             check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"contour.py at {BEFORE_FIX} is not available")
    path = tmp_path / "contour_before_fix.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("contour_before_fix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_map_output_unchanged(tmp_path):
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.05, (96, 96, 96)).astype(np.float32)
    m[30:60, 30:60, 30:60] += 1.0
    with mrcfile.new(tmp_path / "small.mrc") as f:
        f.set_data(m)
        f.voxel_size = 1.0
    before = _before_fix_module(tmp_path)
    got = _run(contour, tmp_path / "small.mrc", tmp_path / "now")
    want = _run(before, tmp_path / "small.mrc", tmp_path / "before")
    assert got == want  # conservative contour and masked percentage
    # compare voxels and grid, not file bytes: mrcfile stamps each new file's label with the time
    for name in ("prot_mask.mrc", "prot_mask_aggressive.mrc"):
        with mrcfile.open(tmp_path / "now" / name) as a, mrcfile.open(tmp_path / "before" / name) as b:
            assert np.array_equal(a.data, b.data)  # type: ignore[arg-type]
            assert a.voxel_size == b.voxel_size and a.header.origin == b.header.origin  # type: ignore[union-attr]
