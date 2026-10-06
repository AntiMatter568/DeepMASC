"""Pass/fail of evaluate_refinement_mask under our PRFSC criteria, on small synthetic postprocess star files."""
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from eval_refinement_mask import evaluate_mask3d, evaluate_refinement_mask  # noqa: E402

RES = [10.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.2, 2.0]
UNMASKED = [1.0, 0.95, 0.8, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0]
CORRECTED = [1.0, 0.9, 0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
# masked 0.143 resolution is 5.0 (first shell below 0.143 is 4.0); first shell <= 0 after it is 3.5
MASKED = [1.0, 0.9, 0.6, 0.3, 0.1, -0.1, -0.2, -0.2, -0.2, -0.2]
# phase-randomised zero at 5.0, the masked 0.143 shell
PHASE_RAND = [0.9, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def write_star(tmp_path, masked=MASKED, phase_rand=PHASE_RAND, name="run_postprocess.star"):
    star = Path(tmp_path) / name
    lines = ["", "# version 30001", "", "data_general", "", "_rlnFinalResolution 5.0", "", "data_fsc", "", "loop_",
             "_rlnSpectralIndex #1", "_rlnResolution #2", "_rlnAngstromResolution #3",
             "_rlnFourierShellCorrelationCorrected #4",
             "_rlnFourierShellCorrelationUnmaskedMaps #5",
             "_rlnFourierShellCorrelationMaskedMaps #6",
             "_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps #7"]
    for i, r in enumerate(RES):
        lines.append(f"{i} {1.0 / r:.6f} {r:.6f} {CORRECTED[i]} {UNMASKED[i]} {masked[i]} {phase_rand[i]}")
    star.write_text("\n".join(lines) + "\n")
    return star


def run(tmp_path, masked=MASKED, phase_rand=PHASE_RAND):
    star = write_star(tmp_path, masked, phase_rand)
    return evaluate_refinement_mask(str(star), str(tmp_path / "out"))


def test_pass_on_shell_equality(tmp_path):
    r = run(tmp_path)
    assert r["masked_res_0_143"] == 5.0
    assert r["phase_rand_zero_res"] == 5.0
    assert r["masked_zero_res"] == 3.5
    assert r["prfsc_pass"] is True


def test_fail_when_phase_rand_zero_is_finer(tmp_path):
    pr = [0.9, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = run(tmp_path, phase_rand=pr)
    assert r["phase_rand_zero_res"] == 4.0
    assert r["prfsc_pass"] is False


def test_pass_when_phase_rand_zero_is_coarser(tmp_path):
    pr = [0.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert run(tmp_path, phase_rand=pr)["prfsc_pass"] is True


def test_fail_when_masked_fsc_never_crosses_0143(tmp_path):
    r = run(tmp_path, masked=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15])
    assert r["valid_masked_0_143"] is False
    assert math.isnan(r["masked_zero_res"])
    assert r["prfsc_pass"] is False


def test_fail_when_phase_rand_never_reaches_zero(tmp_path):
    r = run(tmp_path, phase_rand=[0.9, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert r["valid_phase_rand_zero"] is False
    assert r["prfsc_pass"] is False


def test_fail_when_masked_fsc_never_reaches_zero(tmp_path):
    # drops below 0.143 but the tail stays positive to Nyquist
    r = run(tmp_path, masked=[1.0, 0.9, 0.6, 0.3, 0.1, 0.05, 0.05, 0.02, 0.01, 0.01])
    assert r["valid_masked_0_143"] is True
    assert math.isnan(r["masked_zero_res"])
    assert r["prfsc_pass"] is False


def test_masked_zero_at_nyquist_counts(tmp_path):
    # no exception near Nyquist, but a zero at the last shell is still a zero
    r = run(tmp_path, masked=[1.0, 0.9, 0.6, 0.3, 0.1, 0.05, 0.05, 0.02, 0.01, 0.0])
    assert r["masked_zero_res"] == 2.0
    assert r["prfsc_pass"] is True


def test_old_rule_is_reported_only(tmp_path):
    # unmasked 0.5 resolution is 6.0; phase-rand zero 5.0 is finer, so the old rule fails while the new passes
    r = run(tmp_path)
    assert r["unmasked_res_0_5"] == 6.0
    assert r["legacy_criterion_met"] is False
    assert r["prfsc_pass"] is True
    assert "criterion_met" not in r


@pytest.mark.parametrize(
    "phase_rand, verdict",
    [(PHASE_RAND, 1), ([0.9, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0)],
)
def test_relion_wrapper_writes_the_same_verdict(tmp_path, phase_rand, verdict):
    star = write_star(tmp_path, phase_rand=phase_rand)
    out = tmp_path / "job"
    wrapper = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "gtf_relion4_run_eval_refinement_mask.py")
    subprocess.run([sys.executable, wrapper, "-i", str(star), "-o", str(out)], check=True,
                   capture_output=True)
    summary = (out / "mask_evaluation_summary.star").read_text().split("\n")
    names = [ln.split()[0] for ln in summary if ln.startswith("_rln")]
    values = summary[summary.index(next(ln for ln in summary if ln.startswith("_rln"))) + len(names)].split()
    row = dict(zip(names, values))
    assert int(row["_rlnMaskEvaluationPrfscPass"]) == verdict
    assert float(row["_rlnMaskEvaluationMaskedRes0143"]) == 5.0
    assert float(row["_rlnMaskEvaluationMaskedZeroRes"]) == 3.5
    assert "prfsc_pass" in (out / "mask3d_evaluation.csv").read_text()


# Options for clause 1: reference FSC (masked/unmasked), reference threshold (0.143/0.5), margin in shells.
# Hand-worked curves on RES (shell index 0..9). The reference crossing is the last shell at or above the
# threshold before the first drop below it:
#   masked   0.5: first below at shell 3 (0.3)  -> shell 2 = 6.0 A;  0.143: first below at shell 5 (0.1)  -> shell 4 = 4.0 A
#   unmasked 0.5: first below at shell 5 (0.4)  -> shell 4 = 4.0 A;  0.143: first below at shell 8 (0.1)  -> shell 7 = 2.5 A
# phase-randomised zero is at shell 3 (5.0 A); the masked FSC reaches 0 at shell 6 (clause 2 holds throughout).
OPT_UNMASKED = [1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0]
OPT_MASKED = [1.0, 0.9, 0.6, 0.3, 0.2, 0.1, -0.1, -0.1, -0.1, -0.1]
OPT_PHASE_RAND = [0.9, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def opt_data(unmasked=OPT_UNMASKED, masked=OPT_MASKED, phase_rand=OPT_PHASE_RAND):
    return np.array([RES, unmasked, masked, phase_rand, CORRECTED]).T


@pytest.mark.parametrize(
    "fsc, threshold, reference_res, verdict",
    [
        ("masked", 0.143, 4.0, True),     # PR zero at shell 3, reference shell 4
        ("masked", 0.5, 6.0, False),      # reference shell 2, PR zero is finer
        ("unmasked", 0.5, 4.0, True),     # reference shell 4
        ("unmasked", 0.143, 2.5, True),   # reference shell 7
    ],
)
def test_each_reference_picks_its_crossing(fsc, threshold, reference_res, verdict):
    r = evaluate_mask3d(opt_data(), reference_fsc=fsc, reference_threshold=threshold)
    assert r["reference_res"] == reference_res
    assert r["valid_reference"] is True
    assert r["prfsc_pass"] is verdict
    assert (r["reference_fsc"], r["reference_threshold"], r["margin_shells"]) == (fsc, threshold, 0)


def test_defaults_are_masked_0143_no_margin():
    default = evaluate_mask3d(opt_data())
    explicit = evaluate_mask3d(opt_data(), reference_fsc="masked", reference_threshold=0.143, margin_shells=0)
    assert default == explicit
    assert default["reference_res"] == default["masked_res_0_143"] == 4.0
    assert default["valid_reference"] == default["valid_masked_0_143"]
    assert (default["reference_fsc"], default["reference_threshold"], default["margin_shells"]) == ("masked", 0.143, 0)


@pytest.mark.parametrize(
    "fsc, threshold, last_passing_margin",
    [("masked", 0.143, 1), ("unmasked", 0.5, 1), ("unmasked", 0.143, 4), ("masked", 0.5, -1)],
)
def test_margin_moves_the_verdict_exactly_at_the_boundary(fsc, threshold, last_passing_margin):
    # PR zero at shell 3; the reference shell minus 3 is the largest margin that passes
    kw = dict(reference_fsc=fsc, reference_threshold=threshold)
    if last_passing_margin >= 0:
        assert evaluate_mask3d(opt_data(), margin_shells=last_passing_margin, **kw)["prfsc_pass"] is True
    assert evaluate_mask3d(opt_data(), margin_shells=last_passing_margin + 1, **kw)["prfsc_pass"] is False


def test_margin_does_not_change_the_reported_crossings():
    a = evaluate_mask3d(opt_data())
    b = evaluate_mask3d(opt_data(), margin_shells=3)
    for key in ("masked_res_0_143", "phase_rand_zero_res", "masked_zero_res", "reference_res"):
        assert a[key] == b[key]
    assert b["margin_shells"] == 3


@pytest.mark.parametrize(
    "fsc, threshold, unmasked, masked",
    [
        ("unmasked", 0.5, [1.0] * 9 + [0.6], OPT_MASKED),   # unmasked never below 0.5
        ("masked", 0.5, OPT_UNMASKED, [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.5, 0.5]),
    ],
)
def test_missing_reference_crossing_fails(fsc, threshold, unmasked, masked):
    data = opt_data(unmasked=unmasked, masked=masked)
    r = evaluate_mask3d(data, reference_fsc=fsc, reference_threshold=threshold)
    assert r["valid_reference"] is False
    assert r["prfsc_pass"] is False


def test_clause_2_is_unchanged_by_the_options():
    masked_positive = [1.0, 0.9, 0.6, 0.3, 0.2, 0.1, 0.05, 0.05, 0.02, 0.01]
    r = evaluate_mask3d(opt_data(masked=masked_positive), reference_fsc="unmasked", reference_threshold=0.143)
    assert r["valid_reference"] is True
    assert r["valid_masked_zero"] is False
    assert r["prfsc_pass"] is False


@pytest.mark.parametrize(
    "kw",
    [dict(reference_fsc="corrected"), dict(reference_threshold=0.3), dict(margin_shells=-1), dict(margin_shells=1.5)],
)
def test_invalid_option_values_are_rejected(kw):
    with pytest.raises(ValueError):
        evaluate_mask3d(opt_data(), **kw)


def test_evaluate_refinement_mask_passes_options_and_records_them(tmp_path):
    star = write_star(tmp_path)
    r = evaluate_refinement_mask(str(star), str(tmp_path / "out"), reference_fsc="unmasked",
                                 reference_threshold=0.5, margin_shells=1)
    assert r["reference_res"] == 6.0 and r["margin_shells"] == 1
    csv = next((tmp_path / "out").glob("*.csv")).read_text()
    assert "reference_fsc" in csv and "unmasked" in csv and "margin_shells" in csv


def test_relion_wrapper_options_change_the_verdict_and_are_recorded(tmp_path):
    # default fixture: masked 0.143 shell 5.0 (shell 3), PR zero shell 3 -> margin 1 fails
    star = write_star(tmp_path)
    wrapper = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "gtf_relion4_run_eval_refinement_mask.py")
    out = tmp_path / "job"
    subprocess.run([sys.executable, wrapper, "-i", str(star), "-o", str(out), "--reference_fsc", "masked",
                    "--reference_threshold", "0.143", "--margin_shells", "1"], check=True, capture_output=True)
    summary = (out / "mask_evaluation_summary.star").read_text().split("\n")
    names = [ln.split()[0] for ln in summary if ln.startswith("_rln")]
    values = summary[summary.index(next(ln for ln in summary if ln.startswith("_rln"))) + len(names)].split()
    row = dict(zip(names, values))
    assert int(row["_rlnMaskEvaluationPrfscPass"]) == 0
    assert row["_rlnMaskEvaluationReferenceFsc"] == "masked"
    assert float(row["_rlnMaskEvaluationReferenceThreshold"]) == 0.143
    assert int(row["_rlnMaskEvaluationMarginShells"]) == 1
