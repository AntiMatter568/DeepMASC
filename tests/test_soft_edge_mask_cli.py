"""Command line of soft_edge_mask.py: units, cryoSPARC presets, refusals, box-face warning, header."""
import os
import subprocess
import sys

import mrcfile
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from soft_edge_mask import cryosparc_soft_edge_px, main, write_soft_mask  # noqa: E402


def _blob(shape=(24, 24, 24)):
    m = np.zeros(shape, np.float32)
    m[10:14, 10:14, 10:14] = 1
    return m


def _write(path, data, voxel=1.5, origin=(2.0, 3.0, 4.0)):
    with mrcfile.new(path, overwrite=True) as f:
        f.set_data(data)
        f.voxel_size = voxel
        f.header.origin = origin
    with mrcfile.open(path, mode="r+") as f:
        f.header.nxstart, f.header.nystart, f.header.nzstart = 1, 2, 3


def _data(path):
    with mrcfile.open(path) as f:
        return np.array(f.data)


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "in.mrc"
    _write(p, _blob())
    return p


def test_cli_writes_the_same_mask_as_write_soft_mask(src, tmp_path):
    out = tmp_path / "out.mrc"
    assert main(["-i", str(src), "-o", str(out), "--width_px", "3.5", "--extend_px", "2"]) == 0
    write_soft_mask(src, tmp_path / "ref.mrc", 3.5, extend=2)
    assert np.array_equal(_data(out), _data(tmp_path / "ref.mrc"))


def test_ini_threshold_is_passed_through(tmp_path):
    p = tmp_path / "in.mrc"
    m = _blob()
    m[2, 2, 2] = 0.3
    _write(p, m)
    main(["-i", str(p), "-o", str(tmp_path / "lo.mrc"), "--width_px", "2"])
    main(["-i", str(p), "-o", str(tmp_path / "hi.mrc"), "--width_px", "2", "--ini_threshold", "0.5"])
    assert _data(tmp_path / "lo.mrc")[2, 2, 2] == 1.0
    assert _data(tmp_path / "hi.mrc")[2, 2, 2] == 0.0


def test_angstrom_and_pixel_widths_give_the_same_mask(src, tmp_path):
    # voxel 1.5 A: 6.0 A = 4 px, 3.0 A = 2 px
    main(["-i", str(src), "-o", str(tmp_path / "a.mrc"), "--width_A", "6.0", "--extend_A", "3.0"])
    main(["-i", str(src), "-o", str(tmp_path / "p.mrc"), "--width_px", "4", "--extend_px", "2"])
    assert np.array_equal(_data(tmp_path / "a.mrc"), _data(tmp_path / "p.mrc"))


def test_fractional_pixels_are_kept(src, tmp_path):
    # 5.25 A / 1.5 A/px = 3.5 px
    main(["-i", str(src), "-o", str(tmp_path / "a.mrc"), "--width_A", "5.25"])
    main(["-i", str(src), "-o", str(tmp_path / "p.mrc"), "--width_px", "3.5"])
    assert np.array_equal(_data(tmp_path / "a.mrc"), _data(tmp_path / "p.mrc"))


def test_cryosparc_px_values_by_hand():
    # voxel 1.5 A: v4 6 A + 6 A = 4 px + 4 px; v5 R = 4.5 A: 2R = 9 A = 6 px, 3R = 13.5 A = 9 px
    assert cryosparc_soft_edge_px("cryosparc_v4", 1.5, None) == (4.0, 4.0)
    assert cryosparc_soft_edge_px("cryosparc_v5", 1.5, 4.5) == (6.0, 9.0)


def test_v4_preset_equals_explicit_values(src, tmp_path):
    main(["-i", str(src), "-o", str(tmp_path / "a.mrc"), "--preset", "cryosparc_v4"])
    main(["-i", str(src), "-o", str(tmp_path / "b.mrc"), "--width_px", "4", "--extend_px", "4"])
    assert np.array_equal(_data(tmp_path / "a.mrc"), _data(tmp_path / "b.mrc"))


def test_v5_preset_equals_explicit_values(src, tmp_path):
    main(["-i", str(src), "-o", str(tmp_path / "a.mrc"), "--preset", "cryosparc_v5", "--resolution", "4.5"])
    main(["-i", str(src), "-o", str(tmp_path / "b.mrc"), "--width_px", "9", "--extend_px", "6"])
    assert np.array_equal(_data(tmp_path / "a.mrc"), _data(tmp_path / "b.mrc"))


@pytest.mark.parametrize("extra", [
    ["--preset", "cryosparc_v5"],                                        # v5 without R
    ["--width_px", "3", "--width_A", "4.5"],                             # two units for the width
    ["--width_px", "3", "--extend_px", "1", "--extend_A", "1.5"],        # two units for the padding
    ["--preset", "cryosparc_v4", "--width_px", "3"],                     # preset with explicit width
    ["--preset", "cryosparc_v4", "--extend_A", "3"],                     # preset with explicit padding
    ["--resolution", "4.5", "--width_px", "3"],                          # resolution without v5
    [],                                                                  # no width at all
    ["--width_px", "-1"],                                                # non-positive width
])
def test_refusals_write_nothing(src, tmp_path, extra):
    out = tmp_path / "out.mrc"
    with pytest.raises(SystemExit) as e:
        main(["-i", str(src), "-o", str(out)] + extra)
    assert e.value.code != 0
    assert not out.exists()


def test_box_face_warning(tmp_path, capsys):
    p = tmp_path / "in.mrc"
    _write(p, _blob())
    main(["-i", str(p), "-o", str(tmp_path / "o.mrc"), "--width_px", "2"])
    assert "WARNING" not in capsys.readouterr().out
    main(["-i", str(p), "-o", str(tmp_path / "o.mrc"), "--width_px", "11"])  # faces are 10 px from the blob
    assert "WARNING" in capsys.readouterr().out


def test_summary_reports_width_padding_voxel_and_flag(src, tmp_path, capsys):
    main(["-i", str(src), "-o", str(tmp_path / "o.mrc"), "--width_A", "6", "--extend_A", "3"])
    out = capsys.readouterr().out
    assert "width 4 px (6 A)" in out
    assert "padding 2 px (3 A)" in out
    assert "voxel size 1.5 A" in out
    assert "touches box face: no" in out


def test_header_is_kept(src, tmp_path):
    main(["-i", str(src), "-o", str(tmp_path / "o.mrc"), "--width_px", "3"])
    with mrcfile.open(tmp_path / "o.mrc") as f:
        assert f.data.dtype == np.float32
        assert np.isclose(f.voxel_size.x, 1.5)
        assert tuple(float(v) for v in f.header.origin.tolist()) == (2.0, 3.0, 4.0)
        assert (int(f.header.nxstart), int(f.header.nystart), int(f.header.nzstart)) == (1, 2, 3)
        assert (int(f.header.mapc), int(f.header.mapr), int(f.header.maps)) == (1, 2, 3)


def test_script_runs_as_a_command(src, tmp_path):
    out = tmp_path / "o.mrc"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "soft_edge_mask.py"), "-i", str(src), "-o", str(out),
                        "--width_px", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()


WRAPPER = os.path.join(ROOT, "gtf_relion4_run_soft_edge_mask.py")


def test_relion_wrapper_writes_mask_star_and_success_marker(src, tmp_path):
    job = tmp_path / "External" / "job001"
    r = subprocess.run([sys.executable, WRAPPER, "-i", str(src), "-o", str(job), "--width_A", "6",
                        "--extend_A", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    write_soft_mask(src, tmp_path / "ref.mrc", 4, extend=2)
    assert np.array_equal(_data(job / "soft_mask.mrc"), _data(tmp_path / "ref.mrc"))
    assert (job / "RELION_JOB_EXIT_SUCCESS").exists()
    assert not (job / "RELION_JOB_EXIT_FAILURE").exists()
    nodes = (job / "RELION_OUTPUT_NODES.star").read_text()
    assert f"{job / 'soft_mask.mrc'} Mask3D.mrc" in nodes
    assert f"{job / 'soft_edge_mask_summary.star'} LogFile.star" in nodes
    star = (job / "soft_edge_mask_summary.star").read_text().split("\n")
    names = [ln.split()[0] for ln in star if ln.startswith("_rln")]
    values = star[star.index(next(ln for ln in star if ln.startswith("_rln"))) + len(names)].split()
    row = dict(zip(names, values))
    assert float(row["_rlnSoftEdgeWidthPixel"]) == 4.0
    assert float(row["_rlnSoftEdgePaddingPixel"]) == 2.0
    assert float(row["_rlnSoftEdgeWidthAngstrom"]) == 6.0
    assert float(row["_rlnSoftEdgePaddingAngstrom"]) == 3.0
    assert float(row["_rlnSoftEdgeVoxelSize"]) == 1.5
    assert int(row["_rlnSoftEdgeTouchesBox"]) == 0


def test_relion_wrapper_preset_and_failure_marker(src, tmp_path):
    ok = tmp_path / "ok"
    r = subprocess.run([sys.executable, WRAPPER, "-i", str(src), "-o", str(ok), "--preset", "cryosparc_v5",
                        "--resolution", "4.5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    write_soft_mask(src, tmp_path / "ref.mrc", 9, extend=6)
    assert np.array_equal(_data(ok / "soft_mask.mrc"), _data(tmp_path / "ref.mrc"))
    bad = tmp_path / "bad"
    r = subprocess.run([sys.executable, WRAPPER, "-i", str(src), "-o", str(bad), "--preset", "cryosparc_v5"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert (bad / "RELION_JOB_EXIT_FAILURE").exists()
    assert not (bad / "RELION_JOB_EXIT_SUCCESS").exists()
