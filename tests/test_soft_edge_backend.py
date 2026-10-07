"""EDT backend selection: scipy default, Taichi optional, clear failure or printed fallback when it is missing."""
import os
import subprocess
import sys

import mrcfile
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import soft_edge_mask as sem  # noqa: E402
import optimal_soft_edge_relion as ose  # noqa: E402
from soft_edge_cases import shapes  # noqa: E402
from test_soft_edge_reuse import never_passing, fake_postprocess_fail, run_search  # noqa: E402,F401


@pytest.fixture
def no_taichi(monkeypatch):
    monkeypatch.setitem(sys.modules, "soft_edge_taichi", None)  # `import soft_edge_taichi` raises ImportError


@pytest.fixture
def broken_taichi(monkeypatch):
    """A Taichi module that is importable but whose initialisation fails (no usable device, driver error)."""
    import types
    mod = types.ModuleType("soft_edge_taichi")

    def init(arch="auto", threads=None):
        raise RuntimeError("no device")
    mod.init = init
    monkeypatch.setitem(sys.modules, "soft_edge_taichi", mod)


def test_default_backend_is_scipy():
    assert sem.DEFAULT_EDT_BACKEND == "scipy" and sem.DEFAULT_TAICHI_ARCH == "auto"


@pytest.mark.parametrize("fixture", ["no_taichi", "broken_taichi"])
def test_explicit_taichi_without_taichi_fails_clearly(fixture, request):
    request.getfixturevalue(fixture)
    with pytest.raises(sem.EdtBackendError, match="Taichi"):
        sem.capped_squared_distance(shapes()["sphere"], edt_backend="taichi")


@pytest.mark.parametrize("fixture", ["no_taichi", "broken_taichi"])
def test_auto_falls_back_to_scipy_with_a_note(fixture, request, capsys):
    request.getfixturevalue(fixture)
    data = shapes()["sphere"]
    _, r2 = sem.capped_squared_distance(data, extend=2.0, edt_backend="auto")
    _, ref = sem.capped_squared_distance(data, extend=2.0)
    assert np.array_equal(r2, ref)
    out = capsys.readouterr().out
    assert "Taichi" in out and "scipy" in out


def test_unknown_backend_or_arch_is_refused():
    with pytest.raises(ValueError):
        sem.capped_squared_distance(shapes()["sphere"], edt_backend="numba")
    with pytest.raises(ValueError):
        sem.capped_squared_distance(shapes()["sphere"], edt_backend="taichi", taichi_arch="tpu")


def test_cli_explicit_taichi_without_taichi_exits_with_message(no_taichi, tmp_path, capsys):
    src = tmp_path / "in.mrc"
    with mrcfile.new(src) as f:
        f.set_data(shapes()["sphere"])
        f.voxel_size = 1.5
    with pytest.raises(SystemExit) as e:
        sem.main(["-i", str(src), "-o", str(tmp_path / "o.mrc"), "--width_px", "5", "--edt_backend", "taichi"])
    assert e.value.code != 0
    assert "Taichi" in capsys.readouterr().err
    assert not (tmp_path / "o.mrc").exists()


def test_cli_auto_without_taichi_still_writes_the_mask(no_taichi, tmp_path):
    src = tmp_path / "in.mrc"
    with mrcfile.new(src) as f:
        f.set_data(shapes()["sphere"])
        f.voxel_size = 1.5
    sem.main(["-i", str(src), "-o", str(tmp_path / "a.mrc"), "--width_px", "5", "--edt_backend", "auto"])
    sem.main(["-i", str(src), "-o", str(tmp_path / "b.mrc"), "--width_px", "5"])
    with mrcfile.open(tmp_path / "a.mrc") as a, mrcfile.open(tmp_path / "b.mrc") as b:
        assert np.array_equal(a.data, b.data)


def test_search_rejects_taichi_with_the_relion_engine(never_passing, tmp_path):
    with pytest.raises(ValueError, match="engine"):
        run_search(never_passing, tmp_path / "out", engine="relion", edt_backend="taichi")


def test_search_with_explicit_taichi_missing_fails_before_any_width(never_passing, tmp_path, no_taichi,
                                                                    fake_postprocess_fail):
    with pytest.raises(sem.EdtBackendError, match="Taichi"):
        run_search(never_passing, tmp_path / "out", edt_backend="taichi")


def test_search_auto_without_taichi_matches_scipy(never_passing, tmp_path, no_taichi, fake_postprocess_fail):
    a = run_search(never_passing, tmp_path / "a", edt_backend="auto")
    b = run_search(never_passing, tmp_path / "b")
    with mrcfile.open(a["optimal_mask_path"]) as fa, mrcfile.open(b["optimal_mask_path"]) as fb:
        assert np.array_equal(fa.data, fb.data)
    assert a["edt_backend"] == "auto" and b["edt_backend"] == "scipy"


@pytest.mark.parametrize("script", ["gtf_relion4_run_soft_edge_mask.py", "gtf_relion4_run_optimal_soft_edge.py"])
def test_relion_wrappers_expose_the_backend_options(script):
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), "--help"], capture_output=True, text=True,
                       cwd=ROOT)
    assert "--edt_backend" in r.stdout and "--taichi_arch" in r.stdout, r.stdout + r.stderr
