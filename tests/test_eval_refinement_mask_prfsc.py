"""Pass/fail of evaluate_refinement_mask under our PRFSC criteria, on small synthetic postprocess star files."""
import math
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_refinement_mask import evaluate_refinement_mask  # noqa: E402

RES = [10.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.2, 2.0]
UNMASKED = [1.0, 0.95, 0.8, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0]
CORRECTED = [1.0, 0.9, 0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
# masked 0.143 resolution is 5.0 (first shell below 0.143 is 4.0); first shell <= 0 after it is 3.5
MASKED = [1.0, 0.9, 0.6, 0.3, 0.1, -0.1, -0.2, -0.2, -0.2, -0.2]
# phase-randomised zero at 5.0, the masked 0.143 shell
PHASE_RAND = [0.9, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def write_star(tmp_path, masked=MASKED, phase_rand=PHASE_RAND):
    star = tmp_path / "run_postprocess.star"
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
