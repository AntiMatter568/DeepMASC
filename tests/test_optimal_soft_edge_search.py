"""The soft-edge search returns the narrowest passing width; RELION calls are replaced by fakes."""
import json
import os
import sys

import mrcfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optimal_soft_edge_relion as ose  # noqa: E402
from test_eval_refinement_mask_prfsc import PHASE_RAND, write_star  # noqa: E402

FINER = [0.9, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]    # phase-randomised zero finer than masked 0.143


@pytest.fixture
def inputs(tmp_path):
    """A 24-voxel box with a 6-voxel cube 9 voxels from every face: a 5 px edge stays off the box, 10 px reaches it."""
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
def fake_postprocess(monkeypatch):
    """relion_postprocess replaced by a star writer: the width in the mask file name decides the verdict."""
    calls = []

    def run(widths_that_pass):
        def fake(mask, half1, out_root, angpix, relion_bin=None):
            width = int(os.path.basename(os.path.dirname(mask)).split("_")[-1])
            calls.append(width)
            pr = PHASE_RAND if width in widths_that_pass else FINER
            write_star(os.path.dirname(out_root), phase_rand=pr, name=os.path.basename(out_root) + ".star")
            return True
        monkeypatch.setattr(ose, "run_postprocess", fake)
        return calls
    return run


def search(inputs, tmp_path, **kw):
    return ose.run_soft_edge_search(str(inputs / "emd_1_mask.mrc"), str(tmp_path / "out"),
                                    str(inputs / "emd_1_half1.map"), n_threads=1, **kw)


def test_stops_at_the_first_passing_width(inputs, tmp_path, fake_postprocess):
    calls = fake_postprocess({5, 10})
    r = search(inputs, tmp_path)
    assert r["optimal_soft_edge_width"] == 5 and r["prfsc_pass"] is True
    assert r["stop_reason"] == "pass" and r["no_passing_width"] is False
    assert calls == [5]


def test_narrowest_width_when_a_later_one_passes(inputs, tmp_path, fake_postprocess):
    # 5 fails and 10 passes, although 10 already reaches the box face
    fake_postprocess({10})
    r = search(inputs, tmp_path)
    assert [w["width"] for w in r["widths_tried"]] == [5, 10]
    assert r["optimal_soft_edge_width"] == 10 and r["prfsc_pass"] is True and r["stop_reason"] == "pass"


def test_never_passing_mask_stops_at_the_box_face(inputs, tmp_path, fake_postprocess):
    calls = fake_postprocess(set())
    r = search(inputs, tmp_path)
    assert calls == [5, 10]
    assert r["optimal_soft_edge_width"] == 10
    assert r["prfsc_pass"] is False and r["no_passing_width"] is True and r["stop_reason"] == "box_edge"
    assert [w["prfsc_pass"] for w in r["widths_tried"]] == [False, False]
    assert r["widths_tried"][1]["touches_box"] is True


def test_width_larger_than_the_box_is_not_tried(inputs, tmp_path, fake_postprocess):
    with mrcfile.new(inputs / "emd_1_mask.mrc", overwrite=True) as f:
        f.set_data(np.zeros((8, 8, 8), np.float32))        # empty mask never reaches a face
    calls = fake_postprocess(set())
    r = search(inputs, tmp_path)
    assert calls == [5]
    assert r["stop_reason"] == "box_size" and r["no_passing_width"] is True


def test_resume_skips_widths_already_evaluated(inputs, tmp_path, fake_postprocess):
    calls = fake_postprocess({10})
    first = search(inputs, tmp_path)
    again = search(inputs, tmp_path)
    assert calls == [5, 10]
    assert again["optimal_soft_edge_width"] == first["optimal_soft_edge_width"] == 10
    assert again["widths_tried"] == first["widths_tried"]


def test_resume_continues_after_the_last_evaluated_width(inputs, tmp_path, fake_postprocess):
    calls = fake_postprocess(set())
    ose.run_soft_edge_search(str(inputs / "emd_1_mask.mrc"), str(tmp_path / "out"),
                             str(inputs / "emd_1_half1.map"), n_threads=1)
    out = tmp_path / "out"
    os.remove(out / "soft_edge_10" / "cell.json")          # as if the run stopped before width 10 finished
    search(inputs, tmp_path)
    assert calls == [5, 10, 10]


def test_summary_lists_every_width_with_resolutions(inputs, tmp_path, fake_postprocess):
    fake_postprocess({10})
    r = search(inputs, tmp_path)
    summary = json.loads((tmp_path / "out" / "emd_1_mask_optimal_mask_summary.json").read_text())
    assert summary["optimal_soft_edge_width"] == 10 and summary["prfsc_pass"] is True
    assert summary["stop_reason"] == "pass" and summary["extend_inimask"] == 0
    row = summary["widths_tried"][0]
    assert row["width"] == 5 and row["prfsc_pass"] is False
    assert row["masked_res_0_143"] == 5.0 and row["phase_rand_zero_res"] == 4.0 and row["masked_zero_res"] == 3.5
    assert (tmp_path / "out" / "emd_1_mask_all_parameter_results.csv").exists()
    assert r["optimal_masked_zero_res"] == 3.5


def test_only_the_chosen_soft_mask_is_kept(inputs, tmp_path, fake_postprocess):
    fake_postprocess({10})
    r = search(inputs, tmp_path)
    kept = sorted(str(p.relative_to(tmp_path / "out")) for p in (tmp_path / "out").rglob("*.mrc"))
    assert kept == [os.path.relpath(r["optimal_mask_path"], tmp_path / "out")]


def test_extension_option_pads_the_mask(inputs, tmp_path, fake_postprocess):
    fake_postprocess({5})
    r = search(inputs, tmp_path, extend_inimask=2)
    with mrcfile.open(r["optimal_mask_path"]) as f:
        assert (np.asarray(f.data) == 1).sum() > 6 ** 3
    assert r["optimal_extend_inimask"] == 2
