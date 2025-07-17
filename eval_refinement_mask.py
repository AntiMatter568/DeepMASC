import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import starfile
import os
import pandas as pd


def parse_star_file(filename):
    df = starfile.read(filename)
    fsc_df = df["fsc"]
    fsc_df_data = fsc_df[
        [
            "rlnAngstromResolution",
            "rlnFourierShellCorrelationUnmaskedMaps",
            "rlnFourierShellCorrelationMaskedMaps",
            "rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps",
            "rlnFourierShellCorrelationCorrected",
        ]
    ]
    return fsc_df_data.to_numpy()


def evaluate_mask3d(data):
    """Evaluate if the Mask3D meets the criteria of Method C"""
    # Extract data columns
    resolution = data[:, 0]
    unmasked_fsc = data[:, 1]
    masked_fsc = data[:, 2]
    phase_rand_fsc = data[:, 3]
    corrected_fsc = data[:, 4]

    valid = True
    valid_fsc_0_5 = True
    valid_fsc_0_143 = True
    valid_phase_rand_zero = True

    # Find resolution at FSC = 0.5 without any Mask3D (unmasked)
    unmasked_fsc_0_5_idx = np.where(unmasked_fsc < 0.5)[0]
    if len(unmasked_fsc_0_5_idx) == 0:
        print("Warning: FSC without mask never drops below 0.5")
        unmasked_res_0_5 = resolution[-1]
        valid = False
        valid_fsc_0_5 = False
    else:
        unmasked_res_0_5_idx = unmasked_fsc_0_5_idx[0] - 1
        unmasked_res_0_5 = resolution[unmasked_res_0_5_idx]

    # Find first zero crossing of phase randomized FSC
    phase_rand_zero_idx = np.where(phase_rand_fsc <= 0)[0]
    if len(phase_rand_zero_idx) == 0:
        print("Warning: Phase randomized FSC never crosses zero")
        phase_rand_zero_res = resolution[-1]
        valid = False
        valid_phase_rand_zero = False
    else:
        phase_rand_zero_idx = phase_rand_zero_idx[0]
        phase_rand_zero_res = resolution[phase_rand_zero_idx]

    # Find resolution at FSC = 0.143 with Mask3D (corrected)
    corrected_fsc_0_143_idx = np.where(corrected_fsc < 0.143)[0]
    if len(corrected_fsc_0_143_idx) == 0:
        print("Warning: Corrected FSC never drops below 0.143")
        corrected_res_0_143 = resolution[-1]
        valid = False
        valid_fsc_0_143 = False
    else:
        corrected_fsc_0_143_idx = corrected_fsc_0_143_idx[0] - 1
        corrected_res_0_143 = resolution[corrected_fsc_0_143_idx]

    # Check if the mask meets criteria
    # print("\nResults:")
    # print(f"Resolution (FSC=0.5) without mask: {unmasked_res_0_5:.3f} Å (1/{unmasked_res_0_5:.3f} Å⁻¹)")
    # print(f"First zero crossing of phase randomized FSC: {phase_rand_zero_res:.3f} Å (1/{phase_rand_zero_res:.3f} Å⁻¹)")
    # print(f"Resolution (FSC=0.143) with corrected FSC: {corrected_res_0_143:.3f} Å (1/{corrected_res_0_143:.3f} Å⁻¹)")

    # Criterion check
    criterion_met = phase_rand_zero_res >= unmasked_res_0_5

    # if criterion_met:
    #     print("\n✓ PASS: The first zero crossing of the Phase Randomized FSC Curve is LOWER than")
    #     print("  the FSC resolution with 0.5 criteria without any Mask3D.")
    #     print(f"  ({phase_rand_zero_res:.3f} Å > {unmasked_res_0_5:.3f} Å)")
    # else:
    #     print("\n✗ FAIL: The first zero crossing of the Phase Randomized FSC Curve is HIGHER than")
    #     print("  the FSC resolution with 0.5 criteria without any Mask3D.")
    #     print(f"  ({phase_rand_zero_res:.3f} Å ≤ {unmasked_res_0_5:.3f} Å)")

    return {
        "unmasked_res_0_5": unmasked_res_0_5,
        "phase_rand_zero_res": phase_rand_zero_res,
        "corrected_res_0_143": corrected_res_0_143,
        "criterion_met": criterion_met,
        "valid": valid,
        "valid_fsc_0_5": valid_fsc_0_5,
        "valid_fsc_0_143": valid_fsc_0_143,
        "valid_phase_rand_zero": valid_phase_rand_zero,
    }


def plot_fsc_curves(data, results, filename, save_dir):
    """Plot the FSC curves for visualization with resolution reciprocal (1/Å) on the x-axis"""
    resolution = data[:, 0]
    unmasked_fsc = data[:, 1]
    masked_fsc = data[:, 2]
    phase_rand_fsc = data[:, 3]
    corrected_fsc = data[:, 4]

    # Convert resolution in Å to resolution reciprocal in 1/Å
    resolution_reciprocal = 1.0 / resolution

    # Set seaborn style
    sns.set(style="whitegrid")

    # Create figure with seaborn styling
    plt.figure(figsize=(10, 6))

    # Create a color palette
    palette = sns.color_palette("deep")

    # Plot FSC curves using seaborn
    sns.lineplot(x=resolution_reciprocal, y=unmasked_fsc, color=palette[0], label="Unmasked FSC")
    sns.lineplot(x=resolution_reciprocal, y=masked_fsc, color=palette[1], label="Masked FSC")
    sns.lineplot(x=resolution_reciprocal, y=phase_rand_fsc, color=palette[2], label="Phase Randomized FSC")
    sns.lineplot(x=resolution_reciprocal, y=corrected_fsc, color=palette[3], label="Corrected FSC")

    # Add horizontal lines at thresholds
    plt.axhline(y=0.5, color=palette[0], linestyle="--", alpha=0.5)
    plt.axhline(y=0.143, color=palette[3], linestyle="--", alpha=0.5)
    plt.axhline(y=0.0, color=palette[2], linestyle="--", alpha=0.5)

    # Add vertical lines at key resolutions (in reciprocal space)
    unmasked_res_0_5_recip = 1.0 / results["unmasked_res_0_5"]
    phase_rand_zero_res_recip = 1.0 / results["phase_rand_zero_res"]
    corrected_res_0_143_recip = 1.0 / results["corrected_res_0_143"]

    plt.axvline(x=unmasked_res_0_5_recip, color=palette[0], linestyle=":", alpha=0.7)
    plt.axvline(x=phase_rand_zero_res_recip, color=palette[2], linestyle=":", alpha=0.7)
    plt.axvline(x=corrected_res_0_143_recip, color=palette[3], linestyle=":", alpha=0.7)

    # Add annotations for key resolutions
    plt.annotate(
        f"{results['unmasked_res_0_5']:.2f} Å (FSC=0.5 unmasked)",
        xy=(unmasked_res_0_5_recip, 0.5),
        xytext=(unmasked_res_0_5_recip + 0.01, 0.6),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        f"{results['phase_rand_zero_res']:.2f} Å (Phase Rand. Zero)",
        xy=(phase_rand_zero_res_recip, 0.0),
        xytext=(phase_rand_zero_res_recip + 0.01, 0.1),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        f"{results['corrected_res_0_143']:.2f} Å (FSC=0.143 corrected)",
        xy=(corrected_res_0_143_recip, 0.143),
        xytext=(corrected_res_0_143_recip + 0.01, 0.25),
        arrowprops=dict(arrowstyle="->"),
    )

    # Add pass/fail status
    status = "PASS" if results["criterion_met"] else "FAIL"
    plt.title(f"FSC Curves Evaluation - {status} - {filename}", fontsize=14)

    plt.xlabel("Resolution (1/Å)", fontsize=12)
    plt.ylabel("Fourier Shell Correlation", fontsize=12)
    plt.legend(loc="upper right")

    # Set x-axis limits in reciprocal space (resolution increases from left to right)
    plt.xlim(0, max(resolution_reciprocal) * 1.1)
    plt.ylim(-0.1, 1.1)

    # Add context to the plot
    sns.despine(left=False, bottom=False)

    plt.tight_layout()
    fig_save_name = f'mask3d_evaluation_{filename.split("/")[-1].split(".")[0]}.png'
    plt.savefig(os.path.join(save_dir, fig_save_name), dpi=300)
    print(f"\nFSC curves plot saved as {os.path.join(save_dir, fig_save_name)}")
    # plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_mask3d.py <star_file> <save_dir>")
        sys.exit(1)

    star_file = sys.argv[1]
    save_dir = sys.argv[2]
    print(f"Evaluating Mask3D using file: {star_file}")

    # Parse star file
    data = parse_star_file(star_file)
    if data is None:
        sys.exit(1)

    # Evaluate mask
    results = evaluate_mask3d(data)

    # save result dict as csv
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(save_dir, "mask3d_evaluation.csv"), index=False)

    # Plot results if valid
    if results["valid"]:
        plot_fsc_curves(data, results, star_file, save_dir)


if __name__ == "__main__":
    main()
