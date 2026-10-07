"""The Taichi EDT backend equals scipy: squared distances exactly, soft masks within float rounding."""
import os
import sys

import mrcfile
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

pytest.importorskip("taichi")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soft_edge_mask as sem  # noqa: E402
from soft_edge_cases import WIDTHS, shapes  # noqa: E402

TOL = 1e-6


@pytest.fixture(scope="module", autouse=True)
def _cpu_init():
    sem.capped_squared_distance(np.ones((3, 3, 3), np.float32), edt_backend="taichi", taichi_arch="cpu")


@pytest.mark.parametrize("name", [n for n in shapes() if n not in ("empty",)])
def test_squared_distances_equal_scipy_exactly(name):
    import soft_edge_taichi as st
    binary = shapes()[name] >= 0.01
    if not binary.any() or binary.all():
        pytest.skip("distance undefined without both a mask and background voxel")
    ref = np.rint(distance_transform_edt(~binary) ** 2).astype(np.int64)
    assert np.array_equal(st.edt_sq(binary).astype(np.int64), ref)


@pytest.mark.parametrize("name", list(shapes()))
@pytest.mark.parametrize("extend", [0.0, 3.5])
def test_soft_masks_within_float_rounding_of_scipy(name, extend):
    data = shapes()[name]
    b0, r0 = sem.capped_squared_distance(data, extend=extend)
    b1, r1 = sem.capped_squared_distance(data, extend=extend, edt_backend="taichi", taichi_arch="cpu")
    assert np.array_equal(b0, b1)
    assert np.array_equal(np.rint(r0), r1)
    for w in WIDTHS:
        a = sem.soft_edge_from_distance(b0, r0, w)
        b = sem.soft_edge_from_distance(b1, r1, w)
        assert np.abs(a - b).max() <= TOL


def test_anisotropic_box_and_cap():
    data = np.zeros((130, 17, 9), np.float32)
    data[3, 2, 1] = 1
    b0, r0 = sem.capped_squared_distance(data)
    b1, r1 = sem.capped_squared_distance(data, edt_backend="taichi", taichi_arch="cpu")
    assert r1.max() == 9999.0 and np.array_equal(np.rint(r0), r1)


def test_cli_taichi_matches_default(tmp_path):
    src = tmp_path / "in.mrc"
    with mrcfile.new(src) as f:
        f.set_data(shapes()["two_blobs"])
        f.voxel_size = 1.5
    sem.main(["-i", str(src), "-o", str(tmp_path / "t.mrc"), "--width_px", "7", "--extend_px", "2",
              "--edt_backend", "taichi", "--taichi_arch", "cpu"])
    sem.main(["-i", str(src), "-o", str(tmp_path / "s.mrc"), "--width_px", "7", "--extend_px", "2"])
    with mrcfile.open(tmp_path / "t.mrc") as a, mrcfile.open(tmp_path / "s.mrc") as b:
        assert np.abs(a.data - b.data).max() <= TOL


def test_explicit_cuda_without_a_gpu_fails_clearly_and_cpu_works():
    import soft_edge_taichi as st
    if st.cuda_available():
        pytest.skip("a CUDA device is present")
    with pytest.raises(Exception, match="(?i)cuda"):
        sem.capped_squared_distance(shapes()["sphere"], edt_backend="taichi", taichi_arch="cuda")
