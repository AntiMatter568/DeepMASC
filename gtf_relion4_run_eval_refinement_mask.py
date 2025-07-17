#!/usr/bin/env python

# ***************************************************************************
#
# Copyright (c) 2022-2024 Structural Biology Research Center,
#                         Institute of Materials Structure Science,
#                         High Energy Accelerator Research Organization (KEK)
#
#
# Authors:   Han Zhu, Toshio Moriya (toshio.moriya@kek.jp)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307  USA
#
# ***************************************************************************
#
#
# This script is to evaluate refinement mask quality using FSC criteria
# It designed to be executed as an External job type in Relion GUI
# Create: 2024/12/19 Han Zhu (KEK, SBRC)
#
# Run with Relion external job (RELION4)
# https://relion.readthedocs.io/en/release-4.0/Reference/Using-RELION.html

# Provide executable in the gui: python /path/to/gtf_relion4_run_eval_refinement_mask.py
# Input FSC star file from PostProcess job
# 
# Outputs for RELION
# - mask3d_evaluation.csv
# - mask3d_evaluation_*.png (FSC curves plot)
# - mask_evaluation_summary.star
# - RELION_JOB_EXIT_SUCCESS
# - RELION_OUTPUT_NODES.star

from __future__ import print_function

"""Import >>>"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import starfile

"""<<< Import"""

"""USAGE >>>"""
print("This script evaluates refinement mask quality using FSC criteria")
"""<<< USAGE"""

"""VARIABLES >>>"""
print("running ...")
parser = argparse.ArgumentParser()
# --in_YYY: YYY is the type of the input node: movies, mics, parts, coords, 3dref, or mask,
parser.add_argument("-i", "--input", "--in_postprocess", type=str, help="RELION requirement! Input PostProcess star file path (relative)")
parser.add_argument("-o", "--output", type=str, help="RELION requirement! Output job directory path (relative)")
parser.add_argument("--plot", type=bool, help="Generate FSC curves plot", default=True)
parser.add_argument("--debug", type=bool, help="Enable debug mode to generate full output", default=False)

args, unknown = parser.parse_known_args()

inargs_postprocess = args.input
outargs_rpath = args.output
enable_plot = args.plot
debug_mode = args.debug

print("[GTF_DEBUG] inargs_postprocess  : %s" % inargs_postprocess)
print("[GTF_DEBUG] outargs_rpath       : %s" % outargs_rpath)
print("[GTF_DEBUG] enable_plot         : %s" % enable_plot)
print("[GTF_DEBUG] debug_mode          : %s" % debug_mode)

"""<<< VARIABLES"""

"""Preparation >>>"""
assert os.path.exists(inargs_postprocess), f"# Logical Error: Input PostProcess STAR file ({inargs_postprocess}) must exist."
input_job_dir_rpath, input_postprocess_file_basename = os.path.split(inargs_postprocess)
print("[GTF_DEBUG] input_job_dir_rpath             : %s" % input_job_dir_rpath)
print("[GTF_DEBUG] input_postprocess_file_basename : %s" % input_postprocess_file_basename)

# Ensure output directory exists
os.makedirs(outargs_rpath, exist_ok=True)
"""<<< Preparation"""


"""Functions >>>"""


def parse_star_file(filename):
    """Parse RELION PostProcess star file to extract FSC data"""
    print(f"[GTF_DEBUG] Parsing star file: {filename}")
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
    print("[GTF_DEBUG] Evaluating mask3D criteria...")
    
    # Extract data columns
    resolution = data[:, 0]
    unmasked_fsc = data[:, 1]
    phase_rand_fsc = data[:, 3]
    corrected_fsc = data[:, 4]

    valid = True
    valid_fsc_0_5 = True
    valid_fsc_0_143 = True
    valid_phase_rand_zero = True

    # Find resolution at FSC = 0.5 without any Mask3D (unmasked)
    unmasked_fsc_0_5_idx = np.where(unmasked_fsc < 0.5)[0]
    if len(unmasked_fsc_0_5_idx) == 0:
        print("[GTF_DEBUG] Warning: FSC without mask never drops below 0.5")
        unmasked_res_0_5 = resolution[-1]
        valid = False
        valid_fsc_0_5 = False
    else:
        unmasked_res_0_5_idx = unmasked_fsc_0_5_idx[0] - 1
        unmasked_res_0_5 = resolution[unmasked_res_0_5_idx]

    # Find first zero crossing of phase randomized FSC
    phase_rand_zero_idx = np.where(phase_rand_fsc <= 0)[0]
    if len(phase_rand_zero_idx) == 0:
        print("[GTF_DEBUG] Warning: Phase randomized FSC never crosses zero")
        phase_rand_zero_res = resolution[-1]
        valid = False
        valid_phase_rand_zero = False
    else:
        phase_rand_zero_idx = phase_rand_zero_idx[0]
        phase_rand_zero_res = resolution[phase_rand_zero_idx]

    # Find resolution at FSC = 0.143 with Mask3D (corrected)
    corrected_fsc_0_143_idx = np.where(corrected_fsc < 0.143)[0]
    if len(corrected_fsc_0_143_idx) == 0:
        print("[GTF_DEBUG] Warning: Corrected FSC never drops below 0.143")
        corrected_res_0_143 = resolution[-1]
        valid = False
        valid_fsc_0_143 = False
    else:
        corrected_fsc_0_143_idx = corrected_fsc_0_143_idx[0] - 1
        corrected_res_0_143 = resolution[corrected_fsc_0_143_idx]

    # Check if the mask meets criteria
    criterion_met = phase_rand_zero_res >= unmasked_res_0_5

    print(f"[GTF_DEBUG] Resolution (FSC=0.5) without mask: {unmasked_res_0_5:.3f} Å")
    print(f"[GTF_DEBUG] First zero crossing of phase randomized FSC: {phase_rand_zero_res:.3f} Å")
    print(f"[GTF_DEBUG] Resolution (FSC=0.143) with corrected FSC: {corrected_res_0_143:.3f} Å")
    
    if criterion_met:
        print("[GTF_DEBUG] ✓ PASS: The first zero crossing of the Phase Randomized FSC Curve is LOWER than")
        print("[GTF_DEBUG]   the FSC resolution with 0.5 criteria without any Mask3D.")
        print(f"[GTF_DEBUG]   ({phase_rand_zero_res:.3f} Å > {unmasked_res_0_5:.3f} Å)")
    else:
        print("[GTF_DEBUG] ✗ FAIL: The first zero crossing of the Phase Randomized FSC Curve is HIGHER than")
        print("[GTF_DEBUG]   the FSC resolution with 0.5 criteria without any Mask3D.")
        print(f"[GTF_DEBUG]   ({phase_rand_zero_res:.3f} Å ≤ {unmasked_res_0_5:.3f} Å)")

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
    print("[GTF_DEBUG] Generating FSC curves plot...")
    
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
    plt.title(f"FSC Curves Evaluation - {status} - {os.path.basename(filename)}", fontsize=14)

    plt.xlabel("Resolution (1/Å)", fontsize=12)
    plt.ylabel("Fourier Shell Correlation", fontsize=12)
    plt.legend(loc="upper right")

    # Set x-axis limits in reciprocal space (resolution increases from left to right)
    plt.xlim(0, max(resolution_reciprocal) * 1.1)
    plt.ylim(-0.1, 1.1)

    # Add context to the plot
    sns.despine(left=False, bottom=False)

    plt.tight_layout()
    fig_save_name = f'mask3d_evaluation_{os.path.basename(filename).split(".")[0]}.png'
    plt.savefig(os.path.join(save_dir, fig_save_name), dpi=300)
    print(f"[GTF_DEBUG] FSC curves plot saved as {os.path.join(save_dir, fig_save_name)}")
    plt.close()  # Close the figure to free memory
    
    return fig_save_name


def create_summary_star_file(results, output_dir):
    """Create a RELION-compatible summary star file"""
    print("[GTF_DEBUG] Creating summary star file...")
    
    summary_file = os.path.join(output_dir, "mask_evaluation_summary.star")
    
    with open(summary_file, "w") as f:
        f.write("\n")
        f.write("# version 30001\n")
        f.write("data_mask_evaluation\n")
        f.write("\n")
        f.write("loop_\n")
        f.write("_rlnMaskEvaluationCriterionMet #1\n")
        f.write("_rlnMaskEvaluationUnmaskedRes05 #2\n")
        f.write("_rlnMaskEvaluationPhaseRandZeroRes #3\n")
        f.write("_rlnMaskEvaluationCorrectedRes0143 #4\n")
        f.write("_rlnMaskEvaluationValid #5\n")
        f.write(f"{int(results['criterion_met'])} {results['unmasked_res_0_5']:.6f} "
                f"{results['phase_rand_zero_res']:.6f} {results['corrected_res_0_143']:.6f} "
                f"{int(results['valid'])}\n")
        f.write("\n")
    
    print(f"[GTF_DEBUG] Summary star file saved as {summary_file}")
    return "mask_evaluation_summary.star"


"""<<< Functions"""

"""Main Processing >>>"""
print("[GTF_DEBUG] Starting mask3D evaluation...")

# Parse star file
try:
    data = parse_star_file(inargs_postprocess)
    print(f"[GTF_DEBUG] Successfully parsed {data.shape[0]} data points from star file")
except Exception as e:
    print(f"[GTF_ERROR] Failed to parse star file: {e}")
    sys.exit(1)

# Evaluate mask
results = evaluate_mask3d(data)

# Save results as CSV
result_df = pd.DataFrame([results])
csv_output = os.path.join(outargs_rpath, "mask3d_evaluation.csv")
result_df.to_csv(csv_output, index=False)
print(f"[GTF_DEBUG] Results saved as CSV: {csv_output}")

# Generate plot if requested and valid
plot_filename = None
if enable_plot and results["valid"]:
    plot_filename = plot_fsc_curves(data, results, inargs_postprocess, outargs_rpath)

# Create summary star file
summary_star_filename = create_summary_star_file(results, outargs_rpath)

print("[GTF_DEBUG] Evaluation completed successfully")
"""<<< Main Processing"""

"""Finishing up >>>"""
print("[GTF_DEBUG] Creating RELION output files...")

# Create RELION_OUTPUT_NODES.star file
relion_output_nodes_star_file = open(os.path.join(outargs_rpath, "RELION_OUTPUT_NODES.star"), "w")
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.write("# version 30001\n")
relion_output_nodes_star_file.write("data_output_nodes\n")
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.write("loop_\n")
relion_output_nodes_star_file.write("_rlnPipeLineNodeName #1 \n")
relion_output_nodes_star_file.write("_rlnPipeLineNodeTypeLabel #2 \n")
relion_output_nodes_star_file.write(f"{os.path.join(outargs_rpath, 'mask3d_evaluation.csv')} Text.txt \n")
relion_output_nodes_star_file.write(f"{os.path.join(outargs_rpath, summary_star_filename)} LogFile.star \n")
if plot_filename:
    relion_output_nodes_star_file.write(f"{os.path.join(outargs_rpath, plot_filename)} Image.png \n")
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.close()

# Create RELION_JOB_EXIT_SUCCESS file
relion_job_exit_status_file = open(os.path.join(outargs_rpath, "RELION_JOB_EXIT_SUCCESS"), "w")
relion_job_exit_status_file.close()

print("[GTF_DEBUG] Done")
"""<<< Finishing up""" 