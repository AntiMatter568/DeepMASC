"""One distance transform per mask serves every width; the soft masks equal the per-width computation bit for bit."""
import os
import sys

import mrcfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soft_edge_mask as sem  # noqa: E402
import optimal_soft_edge_relion as ose  # noqa: E402
from soft_edge_cases import EXTENDS, WIDTHS, legacy_soft_edge, shapes  # noqa: E402
from test_eval_refinement_mask_prfsc import write_star  # noqa: E402

@pytest.mark.parametrize("name", list(shapes()))
@pytest.mark.parametrize("extend", EXTENDS)
def test_every_width_from_one_distance_is_bit_identical(name, extend):
    data = shapes()[name]
    binary, r2 = sem.capped_squared_distance(data, extend=extend)
    for w in WIDTHS:
        shared = sem.soft_edge_from_distance(binary, r2, w)
        assert np.array_equal(shared, legacy_soft_edge(data, w, extend=extend)), (name, extend, w)
        assert np.array_equal(shared, sem.relion_soft_edge(data, w, extend=extend))


def test_distance_is_capped_at_9999_and_empty_mask_is_all_cap():
    data = np.zeros((130, 4, 4), np.float32)
    data[0, 0, 0] = 1
    _, r2 = sem.capped_squared_distance(data)
    assert r2.max() == 9999.0 and r2[0, 0, 0] == 0.0
    _, r2e = sem.capped_squared_distance(np.zeros((5, 5, 5), np.float32))
    assert (r2e == 9999.0).all()


def test_padding_uses_a_second_transform_and_none_without_padding(monkeypatch):
    calls = []
    real = sem.distance_transform_edt
    monkeypatch.setattr(sem, "distance_transform_edt", lambda *a, **k: calls.append(1) or real(*a, **k))
    sem.capped_squared_distance(shapes()["sphere"])
    assert len(calls) == 1
    calls.clear()
    sem.capped_squared_distance(shapes()["sphere"], extend=2.0)
    assert len(calls) == 2


def test_write_soft_mask_still_matches_the_per_width_computation(tmp_path):
    data = shapes()["sphere"]
    src = tmp_path / "m.mrc"
    with mrcfile.new(src) as f:
        f.set_data(data)
    for w in (5, 12):
        out = tmp_path / f"o{w}.mrc"
        sem.write_soft_mask(str(src), str(out), w, extend=2.0)
        with mrcfile.open(out) as f:
            assert np.array_equal(f.data, legacy_soft_edge(data, w, extend=2.0).astype(np.float32))


def test_soft_edge_distance_object_writes_each_width_from_one_transform(tmp_path, monkeypatch):
    data = shapes()["sphere"]
    src = tmp_path / "m.mrc"
    with mrcfile.new(src) as f:
        f.set_data(data)
        f.voxel_size = 1.7
    calls = []
    real = sem.distance_transform_edt
    monkeypatch.setattr(sem, "distance_transform_edt", lambda *a, **k: calls.append(1) or real(*a, **k))
    dist = sem.SoftEdgeDistance(str(src), extend=2.0)
    assert not calls  # lazy: nothing computed before the first width is asked for
    for w in (5, 10, 15):
        out = tmp_path / f"o{w}.mrc"
        dist.write(str(out), w)
        with mrcfile.open(out) as f:
            assert np.array_equal(f.data, legacy_soft_edge(data, w, extend=2.0).astype(np.float32))
            assert float(f.voxel_size.x) == pytest.approx(1.7)
    assert len(calls) == 2


@pytest.fixture
def never_passing(tmp_path):
    """A cube 9 voxels from every face of a 24 box: 5 px stays off the faces, 10 px reaches them (two widths)."""
    d = tmp_path / "in"
    d.mkdir()
    m = np.zeros((24, 24, 24), np.float32)
    m[9:15, 9:15, 9:15] = 1
    with mrcfile.new(d / "emd_1_mask.mrc") as f:
        f.set_data(m)
    for name in ("emd_1_half1.map", "emd_1_half2.map"):
        with mrcfile.new(d / name) as f:
            f.set_data(np.zeros((24, 24, 24), np.float32))
    return d


@pytest.fixture
def fake_postprocess_fail(monkeypatch):
    def fake(mask, half1, out_root, angpix, relion_bin=None):
        from test_optimal_soft_edge_search import FINER
        write_star(os.path.dirname(out_root), phase_rand=FINER, name=os.path.basename(out_root) + ".star")
        return True
    monkeypatch.setattr(ose, "run_postprocess", fake)


def run_search(inputs, out, **kw):
    return ose.run_soft_edge_search(str(inputs / "emd_1_mask.mrc"), str(out), str(inputs / "emd_1_half1.map"),
                                    n_threads=1, **kw)


@pytest.mark.parametrize("extend,n_transforms", [(0, 1), (2, 2)])
def test_search_computes_the_distance_once_for_all_widths(never_passing, tmp_path, fake_postprocess_fail,
                                                          monkeypatch, extend, n_transforms):
    calls = []
    real = sem.distance_transform_edt
    monkeypatch.setattr(sem, "distance_transform_edt", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = run_search(never_passing, tmp_path / "out", extend_inimask=extend)
    assert [c["width"] for c in r["widths_tried"]] == [5, 10] and r["stop_reason"] == "box_edge"
    assert len(calls) == n_transforms
    data = np.asarray(mrcfile.open(never_passing / "emd_1_mask.mrc").data)
    with mrcfile.open(r["optimal_mask_path"]) as f:
        assert np.array_equal(f.data, legacy_soft_edge(data, 10, extend=extend).astype(np.float32))


def test_search_resumed_with_every_width_cached_computes_nothing(never_passing, tmp_path, fake_postprocess_fail,
                                                                 monkeypatch):
    run_search(never_passing, tmp_path / "out")
    calls = []
    real = sem.distance_transform_edt
    monkeypatch.setattr(sem, "distance_transform_edt", lambda *a, **k: calls.append(1) or real(*a, **k))
    run_search(never_passing, tmp_path / "out")
    assert calls == []
