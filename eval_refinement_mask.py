import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import starfile
import os
import pandas as pd


def parse_star_file(filename):
    df = starfile.read(filename)
    fsc_df = df["fsc"]  # type: ignore[index]
    fsc_df_data = fsc_df[  # type: ignore[index]
        [
            "rlnAngstromResolution",
            "rlnFourierShellCorrelationUnmaskedMaps",
            "rlnFourierShellCorrelationMaskedMaps",
            "rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps",
            "rlnFourierShellCorrelationCorrected",
        ]
    ]
    return fsc_df_data.to_numpy()  # type: ignore[union-attr]


def _find_resolution_at_threshold(fsc_values, resolution, threshold, direction="below"):
    """Find the resolution where FSC crosses a threshold.

    Args:
        fsc_values: FSC curve values
        resolution: Resolution values in Angstroms (decreasing = higher resolution)
        threshold: FSC threshold to find crossing for
        direction: "below" for first crossing below threshold, "zero" for first <= 0

    Returns:
        (resolution_value, valid) tuple. If crossing not found, returns (resolution[-1], False).
    """
    if direction == "zero":
        crossing_idx = np.where(fsc_values <= 0)[0]
    else:
        crossing_idx = np.where(fsc_values < threshold)[0]

    if len(crossing_idx) == 0:
        return resolution[-1], False

    if direction == "zero":
        return resolution[crossing_idx[0]], True

    idx = max(crossing_idx[0] - 1, 0)
    return resolution[idx], True


def _find_masked_zero_resolution(masked_fsc, resolution):
    """Find the first shell with masked FSC <= 0 at or after the first shell below 0.143.

    Returns:
        (resolution_value, valid) tuple. If the masked FSC never drops below 0.143, or never
        reaches 0 after that up to Nyquist, returns (nan, False).
    """
    below = np.where(masked_fsc < 0.143)[0]
    if len(below) == 0:
        return np.nan, False
    at_zero = np.where(masked_fsc[below[0] :] <= 0)[0]
    if len(at_zero) == 0:
        return np.nan, False
    return resolution[below[0] + at_zero[0]], True


def _shell_index(resolution, value):
    """Index of the shell with this resolution value."""
    return int(np.flatnonzero(resolution == value)[0])


def evaluate_mask3d(
    data, reference_fsc="masked", reference_threshold=0.143, margin_shells=0
):
    """Evaluate a mask from the FSC curves of a postprocess run.

    Clause 1 of the PRFSC criterion compares the phase randomized zero (first shell at or below 0)
    with a reference crossing; the options choose the reference. The defaults are the paper's
    criteria: masked FSC, 0.143, no margin.

    Args:
        data: array with columns resolution, unmasked, masked, phase randomized, corrected FSC
        reference_fsc: "masked" or "unmasked": the FSC on which the reference crossing is read
        reference_threshold: 0.143 or 0.5: the threshold of the reference crossing
        margin_shells: whole shells (default 0) by which the phase randomized zero must lie
            coarser than the reference crossing
    """
    if reference_fsc not in ("masked", "unmasked"):
        raise ValueError(f"reference_fsc must be 'masked' or 'unmasked', not {reference_fsc!r}")
    if reference_threshold not in (0.143, 0.5):
        raise ValueError(f"reference_threshold must be 0.143 or 0.5, not {reference_threshold!r}")
    if isinstance(margin_shells, bool) or int(margin_shells) != margin_shells or margin_shells < 0:
        raise ValueError(f"margin_shells must be a whole number >= 0, not {margin_shells!r}")
    margin_shells = int(margin_shells)

    resolution = data[:, 0]
    unmasked_fsc = data[:, 1]
    masked_fsc = data[:, 2]
    phase_rand_fsc = data[:, 3]
    corrected_fsc = data[:, 4]

    unmasked_res_0_5, valid_fsc_0_5 = _find_resolution_at_threshold(
        unmasked_fsc, resolution, 0.5
    )
    if not valid_fsc_0_5:
        print("Warning: FSC without mask never drops below 0.5")

    phase_rand_zero_res, valid_phase_rand_zero = _find_resolution_at_threshold(
        phase_rand_fsc, resolution, 0, direction="zero"
    )
    if not valid_phase_rand_zero:
        print("Warning: Phase randomized FSC never crosses zero")

    corrected_res_0_143, valid_corrected_0_143 = _find_resolution_at_threshold(
        corrected_fsc, resolution, 0.143
    )
    if not valid_corrected_0_143:
        print("Warning: Corrected FSC never drops below 0.143")

    unmasked_res_0_143, valid_unmasked_0_143 = _find_resolution_at_threshold(
        unmasked_fsc, resolution, 0.143
    )
    if not valid_unmasked_0_143:
        print("Warning: Unmasked FSC never drops below 0.143")

    masked_res_0_143, valid_masked_0_143 = _find_resolution_at_threshold(
        masked_fsc, resolution, 0.143
    )
    if not valid_masked_0_143:
        print("Warning: Masked FSC never drops below 0.143")

    masked_zero_res, valid_masked_zero = _find_masked_zero_resolution(
        masked_fsc, resolution
    )
    if valid_masked_0_143 and not valid_masked_zero:
        print("Warning: Masked FSC never reaches zero after its 0.143 crossing")

    # |masked_0.143 - corrected_0.143|: smaller = less correction needed = more reliable mask
    correction_magnitude = abs(masked_res_0_143 - corrected_res_0_143)

    beyond_resolution_mask = resolution < corrected_res_0_143
    if np.any(beyond_resolution_mask):
        phase_rand_noise_floor = float(
            np.max(np.abs(phase_rand_fsc[beyond_resolution_mask]))
        )
    else:
        phase_rand_noise_floor = 0.0

    valid = valid_fsc_0_5 and valid_phase_rand_zero and valid_corrected_0_143

    # Reference crossing of clause 1 (default: the masked 0.143 resolution)
    reference_curve = masked_fsc if reference_fsc == "masked" else unmasked_fsc
    reference_res, valid_reference = _find_resolution_at_threshold(
        reference_curve, resolution, reference_threshold
    )
    if not valid_reference:
        print(
            f"Warning: {reference_fsc.capitalize()} FSC never drops below {reference_threshold}"
        )

    # Our PRFSC criteria (defaults): the phase randomized FSC reaches 0 at or before the shell of
    # the masked 0.143 resolution (no margin), and the masked FSC itself reaches 0 beyond its
    # 0.143 crossing, at or before Nyquist. A missing crossing fails. With a margin of x shells
    # the phase randomized zero must lie at least x shells coarser than the reference crossing.
    prfsc_pass = bool(
        valid_reference
        and valid_phase_rand_zero
        and valid_masked_zero
        and _shell_index(resolution, phase_rand_zero_res) + margin_shells
        <= _shell_index(resolution, reference_res)
    )

    # Old rule (Chen et al. 2013 heuristic), reported only; not part of the pass verdict
    legacy_criterion_met = bool(phase_rand_zero_res >= unmasked_res_0_5)

    return {
        "unmasked_res_0_5": unmasked_res_0_5,
        "phase_rand_zero_res": phase_rand_zero_res,
        "corrected_res_0_143": corrected_res_0_143,
        "unmasked_res_0_143": unmasked_res_0_143,
        "masked_res_0_143": masked_res_0_143,
        "correction_magnitude": correction_magnitude,
        "phase_rand_noise_floor": phase_rand_noise_floor,
        "masked_zero_res": masked_zero_res,
        "reference_fsc": reference_fsc,
        "reference_threshold": reference_threshold,
        "margin_shells": margin_shells,
        "reference_res": reference_res,
        "valid_reference": valid_reference,
        "prfsc_pass": prfsc_pass,
        "legacy_criterion_met": legacy_criterion_met,
        "valid": valid,
        "valid_fsc_0_5": valid_fsc_0_5,
        "valid_fsc_0_143": valid_corrected_0_143,
        "valid_unmasked_0_143": valid_unmasked_0_143,
        "valid_masked_0_143": valid_masked_0_143,
        "valid_masked_zero": valid_masked_zero,
        "valid_phase_rand_zero": valid_phase_rand_zero,
    }


def plot_fsc_curves(data, results, filename, save_dir):
    resolution = data[:, 0]
    unmasked_fsc = data[:, 1]
    masked_fsc = data[:, 2]
    phase_rand_fsc = data[:, 3]
    corrected_fsc = data[:, 4]

    resolution_reciprocal = 1.0 / resolution

    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    assert isinstance(ax1, plt.Axes)
    palette = sns.color_palette("deep")

    ax1.plot(
        resolution_reciprocal, unmasked_fsc, color=palette[0], label="Unmasked FSC"
    )
    ax1.plot(resolution_reciprocal, masked_fsc, color=palette[1], label="Masked FSC")
    ax1.plot(
        resolution_reciprocal,
        phase_rand_fsc,
        color=palette[2],
        label="Phase Randomized FSC",
    )
    ax1.plot(
        resolution_reciprocal, corrected_fsc, color=palette[3], label="Corrected FSC"
    )

    ax1.axhline(y=0.5, color=palette[0], linestyle="--", alpha=0.5)  # type: ignore[arg-type]
    ax1.axhline(y=0.143, color=palette[3], linestyle="--", alpha=0.5)  # type: ignore[arg-type]
    ax1.axhline(y=0.0, color=palette[2], linestyle="--", alpha=0.5)  # type: ignore[arg-type]

    unmasked_res_0_5_recip = 1.0 / results["unmasked_res_0_5"]
    phase_rand_zero_res_recip = 1.0 / results["phase_rand_zero_res"]
    corrected_res_0_143_recip = 1.0 / results["corrected_res_0_143"]

    ax1.axvline(x=unmasked_res_0_5_recip, color=palette[0], linestyle=":", alpha=0.7)
    ax1.axvline(x=phase_rand_zero_res_recip, color=palette[2], linestyle=":", alpha=0.7)
    ax1.axvline(x=corrected_res_0_143_recip, color=palette[3], linestyle=":", alpha=0.7)

    ax1.annotate(
        f"{results['unmasked_res_0_5']:.2f} Å (FSC=0.5 unmasked)",
        xy=(unmasked_res_0_5_recip, 0.5),
        xytext=(unmasked_res_0_5_recip + 0.01, 0.6),
        arrowprops=dict(arrowstyle="->"),
    )
    ax1.annotate(
        f"{results['phase_rand_zero_res']:.2f} Å (Phase Rand. Zero)",
        xy=(phase_rand_zero_res_recip, 0.0),
        xytext=(phase_rand_zero_res_recip + 0.01, 0.1),
        arrowprops=dict(arrowstyle="->"),
    )
    ax1.annotate(
        f"{results['corrected_res_0_143']:.2f} Å (FSC=0.143 corrected)",
        xy=(corrected_res_0_143_recip, 0.143),
        xytext=(corrected_res_0_143_recip + 0.01, 0.25),
        arrowprops=dict(arrowstyle="->"),
    )

    x_max = max(resolution_reciprocal) * 1.1
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("Spatial Frequency (1/Å)", fontsize=12)
    ax1.set_ylabel("Fourier Shell Correlation", fontsize=12)
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    resolution_ticks_angstrom = np.array([50, 20, 10, 7, 5, 4, 3.5, 3, 2.5, 2, 1.5])
    resolution_ticks_recip = 1.0 / resolution_ticks_angstrom
    within_range = resolution_ticks_recip <= x_max
    resolution_ticks_angstrom = resolution_ticks_angstrom[within_range]
    resolution_ticks_recip = resolution_ticks_recip[within_range]
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(resolution_ticks_recip)
    ax2.set_xticklabels([f"{r:g}" for r in resolution_ticks_angstrom])
    ax2.set_xlabel("Resolution (Å)", fontsize=12)

    status = "PASS" if results["prfsc_pass"] else "FAIL"
    ax1.set_title(f"FSC Curves Evaluation - {status} - {filename}", fontsize=14, pad=30)

    sns.despine(left=False, bottom=False)
    assert isinstance(fig, plt.Figure)
    fig.tight_layout()
    fig_save_name = f"mask3d_evaluation_{filename.split('/')[-1].split('.')[0]}.png"
    fig.savefig(os.path.join(save_dir, fig_save_name), dpi=300, bbox_inches="tight")
    print(f"\nFSC curves plot saved as {os.path.join(save_dir, fig_save_name)}")
    plt.close(fig)

    return fig_save_name


def evaluate_refinement_mask(
    star_file,
    save_dir,
    reference_fsc="masked",
    reference_threshold=0.143,
    margin_shells=0,
):
    """
    Evaluate refinement mask from a star file.

    Args:
        star_file (str): Path to the star file containing FSC data
        save_dir (str): Directory to save evaluation results and plots
        reference_fsc (str): "masked" (default) or "unmasked": FSC of the reference crossing
        reference_threshold (float): 0.143 (default) or 0.5: threshold of the reference crossing
        margin_shells (int): shells by which the phase randomized zero must precede the reference

    Returns:
        dict: Evaluation results containing resolutions and pass/fail status
    """
    print(f"Evaluating refinement mask using file: {star_file}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Parse star file
    try:
        data = parse_star_file(star_file)
    except Exception as e:
        print(f"Error parsing star file {star_file}: {e}")
        return None

    # Evaluate mask
    results = evaluate_mask3d(
        data,
        reference_fsc=reference_fsc,
        reference_threshold=reference_threshold,
        margin_shells=margin_shells,
    )

    # Save result dict as csv
    result_df = pd.DataFrame([results])  # Wrap in list to create single row DataFrame
    csv_filename = f"mask3d_evaluation_{os.path.basename(star_file).split('.')[0]}.csv"
    result_df.to_csv(os.path.join(save_dir, csv_filename), index=False)

    # Plot results if valid
    if results["valid"]:
        plot_fsc_curves(data, results, star_file, save_dir)

    return results


def add_criterion_arguments(parser):
    """Command-line options for clause 1 of the PRFSC criterion (defaults: our criteria)."""
    parser.add_argument(
        "--reference_fsc",
        choices=["masked", "unmasked"],
        default="masked",
        help="FSC on which the reference crossing is read (default masked)",
    )
    parser.add_argument(
        "--reference_threshold",
        type=float,
        choices=[0.143, 0.5],
        default=0.143,
        help="Threshold of the reference crossing (default 0.143)",
    )
    parser.add_argument(
        "--margin_shells",
        type=int,
        default=0,
        help="Shells by which the phase randomized zero must lie coarser than the reference "
        "crossing (default 0)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a refinement mask from a RELION postprocess star file."
    )
    parser.add_argument("star_file")
    parser.add_argument("save_dir")
    add_criterion_arguments(parser)
    args = parser.parse_args()

    results = evaluate_refinement_mask(
        args.star_file,
        args.save_dir,
        reference_fsc=args.reference_fsc,
        reference_threshold=args.reference_threshold,
        margin_shells=args.margin_shells,
    )

    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
