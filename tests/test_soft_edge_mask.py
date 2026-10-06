"""Distance-transform soft edge against a literal port of RELION's loop and against relion_mask_create."""
import os
import shutil
import subprocess
import sys

import mrcfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soft_edge_mask import relion_soft_edge, touches_box, write_soft_mask  # noqa: E402

RELION_BIN = shutil.which("relion_mask_create") or "/apps/relion/5.0.1-sm86-resolute/bin/relion_mask_create"
needs_relion = pytest.mark.skipif(not os.path.exists(RELION_BIN), reason="relion_mask_create not available")


def _relion_reference(data, w, ini=0.01, extend=0.0):
    """Literal port of RELION 5.0.1 autoMask steps A, B (extension, positive only) and C
    (src/mask.cpp:303-319, 321-372, 446-497), for small arrays."""
    msk = np.where(data >= ini, 1.0, 0.0)
    if extend > 0:
        e = int(np.ceil(extend))
        cp = msk.copy()
        nz, ny, nx = cp.shape
        for k in range(nz):
            for i in range(ny):
                for j in range(nx):
                    if cp[k, i, j] < 0.001:
                        done = False
                        for kp in range(k - e, k + e + 1):
                            for ip in range(i - e, i + e + 1):
                                for jp in range(j - e, j + e + 1):
                                    if 0 <= kp < nz and 0 <= ip < ny and 0 <= jp < nx and cp[kp, ip, jp] > 0.999:
                                        if float((kp - k) ** 2 + (ip - i) ** 2 + (jp - j) ** 2) < extend * extend:
                                            msk[k, i, j] = 1.0
                                            done = True
                                    if done:
                                        break
                                if done:
                                    break
                            if done:
                                break
    out = msk.copy()
    ext = int(np.ceil(w))
    nz, ny, nx = msk.shape
    for k in range(nz):
        for i in range(ny):
            for j in range(nx):
                if msk[k, i, j] < 0.001:
                    min_r2 = 9999.0
                    for kp in range(max(k - ext, 0), min(k + ext, nz - 1) + 1):
                        for ip in range(max(i - ext, 0), min(i + ext, ny - 1) + 1):
                            for jp in range(max(j - ext, 0), min(j + ext, nx - 1) + 1):
                                if msk[kp, ip, jp] > 0.999:
                                    r2 = float((kp - k) ** 2 + (ip - i) ** 2 + (jp - j) ** 2)
                                    if r2 < min_r2:
                                        min_r2 = r2
                    if min_r2 < w * w:
                        out[k, i, j] = 0.5 + 0.5 * np.cos(np.pi * np.sqrt(min_r2) / w)
    return out


def _blob(shape=(14, 13, 12), seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros(shape, np.float32)
    data[4:9, 3:8, 5:10] = 1
    data[0, 0, 0] = 1
    data[rng.random(shape) > 0.985] = 1
    return data


@pytest.mark.parametrize("w", [1, 2, 3, 4.5, 6])
def test_soft_edge_matches_the_relion_loop(w):
    data = _blob(seed=int(w * 10))
    assert np.allclose(relion_soft_edge(data, w), _relion_reference(data, w), atol=1e-12)


@pytest.mark.parametrize("extend, w", [(1, 2), (2, 3), (3, 3), (2.5, 4.5), (4.58, 4.58)])
def test_padded_soft_edge_matches_the_relion_loop(extend, w):
    data = _blob((16, 15, 14), seed=int(extend * 100 + w))
    assert np.allclose(relion_soft_edge(data, w, extend=extend), _relion_reference(data, w, extend=extend), atol=1e-12)


def test_padding_excludes_a_voxel_exactly_at_the_padding_distance():
    data = np.zeros((1, 1, 12), np.float32)
    data[0, 0, 0] = 1
    s = relion_soft_edge(data, 4, extend=3)[0, 0]
    assert np.allclose(s[:3], 1.0)
    assert s[3] == pytest.approx(0.5 + 0.5 * np.cos(np.pi / 4))
    assert s[7] == 0.0


def test_negative_padding_is_refused():
    with pytest.raises(ValueError):
        relion_soft_edge(np.ones((3, 3, 3), np.float32), 2, extend=-1)


def test_threshold_is_at_or_above():
    data = np.zeros((7, 7, 7), np.float32)
    data[3, 3, 3] = 0.01
    assert relion_soft_edge(data, 2)[3, 3, 3] == 1.0


def test_placeholder_beyond_100_px():
    # RELION starts at r2 = 9999, so a voxel with no mask voxel in reach keeps it and, for w > 99.995, is not zero
    data = np.zeros((6, 6, 6), np.float32)
    for w, expect in ((50, 0.0), (120, 0.5 + 0.5 * np.cos(np.pi * np.sqrt(9999) / 120))):
        assert np.allclose(relion_soft_edge(data, w), expect)


def test_real_distances_are_capped_at_the_placeholder():
    data = np.zeros((1, 1, 230), np.float32)
    data[0, 0, 0] = 1
    s = relion_soft_edge(data, 150)
    d = np.arange(230.0)
    expect = np.where(d == 0, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.sqrt(np.minimum(d * d, 9999.0)) / 150))
    assert np.allclose(s[0, 0], expect)


def test_touches_box():
    m = np.zeros((6, 6, 6), np.float32)
    m[2:4, 2:4, 2:4] = 1
    assert not touches_box(m)
    m[5, 3, 3] = 0.1
    assert touches_box(m)


def _write(path, data, voxel=1.3, origin=(2.0, 3.0, 4.0)):
    with mrcfile.new(path, overwrite=True) as f:
        f.set_data(data)
        f.voxel_size = voxel
        f.header.origin = origin


@pytest.mark.parametrize("extend", [0, 2.5])
def test_written_mask_keeps_the_input_geometry(tmp_path, extend):
    src = tmp_path / "in.mrc"
    m = np.zeros((14, 14, 14), np.float32)
    m[6:8, 6:8, 6:8] = 1
    _write(src, m)
    with mrcfile.open(src, mode="r+") as f:
        f.header.nxstart, f.header.nystart, f.header.nzstart = 1, 2, 3
    write_soft_mask(src, tmp_path / "out.mrc", 3, extend=extend)
    with mrcfile.open(tmp_path / "out.mrc") as f:
        assert f.data.dtype == np.float32
        assert np.allclose(f.data, relion_soft_edge(m, 3, extend=extend))
        assert np.isclose(f.voxel_size.x, 1.3)
        assert tuple(float(v) for v in f.header.origin.tolist()) == (2.0, 3.0, 4.0)
        assert (int(f.header.nxstart), int(f.header.nystart), int(f.header.nzstart)) == (1, 2, 3)
        assert (int(f.header.mapc), int(f.header.mapr), int(f.header.maps)) == (1, 2, 3)


@needs_relion
@pytest.mark.parametrize("extend, w", [(0, 3), (0, 4.5), (2, 3), (0, 100), (0, 150), (3, 120)])
def test_distance_engine_equals_relion_mask_create(tmp_path, extend, w):
    src = tmp_path / "in.mrc"
    data = _blob((10, 9, 8))
    _write(src, data, voxel=1.0, origin=(0.0, 0.0, 0.0))
    relion_out = tmp_path / "relion.mrc"
    subprocess.run([RELION_BIN, "--i", str(src), "--o", str(relion_out), "--ini_threshold", "0.01",
                    "--extend_inimask", str(extend), "--width_soft_edge", str(w), "--j", "1"],
                   check=True, capture_output=True)
    write_soft_mask(src, tmp_path / "py.mrc", w, extend=extend)
    with mrcfile.open(relion_out) as a, mrcfile.open(tmp_path / "py.mrc") as b:
        assert np.allclose(a.data, b.data, atol=3e-8, rtol=0)
