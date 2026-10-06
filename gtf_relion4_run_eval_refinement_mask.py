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
# This script is to evaluate refinement mask quality using the PRFSC criteria
# (evaluation code is shared with eval_refinement_mask.py)
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
import pandas as pd

from eval_refinement_mask import (
    add_criterion_arguments,
    evaluate_mask3d,
    parse_star_file,
    plot_fsc_curves,
)

"""<<< Import"""

"""USAGE >>>"""
print("This script evaluates refinement mask quality using the PRFSC criteria")
"""<<< USAGE"""

"""VARIABLES >>>"""
print("running ...")
parser = argparse.ArgumentParser()
# --in_YYY: YYY is the type of the input node: movies, mics, parts, coords, 3dref, or mask,
parser.add_argument(
    "-i",
    "--input",
    "--in_postprocess",
    type=str,
    help="RELION requirement! Input PostProcess star file path (relative)",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="RELION requirement! Output job directory path (relative)",
)
parser.add_argument("--plot", type=bool, help="Generate FSC curves plot", default=True)
parser.add_argument(
    "--debug",
    type=bool,
    help="Enable debug mode to generate full output",
    default=False,
)

add_criterion_arguments(parser)

args, unknown = parser.parse_known_args()

inargs_postprocess = args.input
outargs_rpath = args.output
enable_plot = args.plot
debug_mode = args.debug
reference_fsc = args.reference_fsc
reference_threshold = args.reference_threshold
margin_shells = args.margin_shells

print("[GTF_DEBUG] inargs_postprocess  : %s" % inargs_postprocess)
print("[GTF_DEBUG] outargs_rpath       : %s" % outargs_rpath)
print("[GTF_DEBUG] enable_plot         : %s" % enable_plot)
print("[GTF_DEBUG] debug_mode          : %s" % debug_mode)
print("[GTF_DEBUG] reference_fsc       : %s" % reference_fsc)
print("[GTF_DEBUG] reference_threshold : %s" % reference_threshold)
print("[GTF_DEBUG] margin_shells       : %s" % margin_shells)

"""<<< VARIABLES"""

"""Preparation >>>"""
assert os.path.exists(inargs_postprocess), (
    f"# Logical Error: Input PostProcess STAR file ({inargs_postprocess}) must exist."
)
input_job_dir_rpath, input_postprocess_file_basename = os.path.split(inargs_postprocess)
print("[GTF_DEBUG] input_job_dir_rpath             : %s" % input_job_dir_rpath)
print(
    "[GTF_DEBUG] input_postprocess_file_basename : %s" % input_postprocess_file_basename
)

# Ensure output directory exists
os.makedirs(outargs_rpath, exist_ok=True)
"""<<< Preparation"""


"""Functions >>>"""


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
        f.write("_rlnMaskEvaluationPrfscPass #1\n")
        f.write("_rlnMaskEvaluationMaskedRes0143 #2\n")
        f.write("_rlnMaskEvaluationPhaseRandZeroRes #3\n")
        f.write("_rlnMaskEvaluationMaskedZeroRes #4\n")
        f.write("_rlnMaskEvaluationLegacyCriterionMet #5\n")
        f.write("_rlnMaskEvaluationUnmaskedRes05 #6\n")
        f.write("_rlnMaskEvaluationCorrectedRes0143 #7\n")
        f.write("_rlnMaskEvaluationValid #8\n")
        f.write("_rlnMaskEvaluationReferenceFsc #9\n")
        f.write("_rlnMaskEvaluationReferenceThreshold #10\n")
        f.write("_rlnMaskEvaluationMarginShells #11\n")
        f.write(
            f"{int(results['prfsc_pass'])} {results['masked_res_0_143']:.6f} "
            f"{results['phase_rand_zero_res']:.6f} {results['masked_zero_res']:.6f} "
            f"{int(results['legacy_criterion_met'])} {results['unmasked_res_0_5']:.6f} "
            f"{results['corrected_res_0_143']:.6f} {int(results['valid'])} "
            f"{results['reference_fsc']} {results['reference_threshold']} "
            f"{results['margin_shells']}\n"
        )
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
results = evaluate_mask3d(
    data,
    reference_fsc=reference_fsc,
    reference_threshold=reference_threshold,
    margin_shells=margin_shells,
)

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
relion_output_nodes_star_file = open(
    os.path.join(outargs_rpath, "RELION_OUTPUT_NODES.star"), "w"
)
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.write("# version 30001\n")
relion_output_nodes_star_file.write("data_output_nodes\n")
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.write("loop_\n")
relion_output_nodes_star_file.write("_rlnPipeLineNodeName #1 \n")
relion_output_nodes_star_file.write("_rlnPipeLineNodeTypeLabel #2 \n")
relion_output_nodes_star_file.write(
    f"{os.path.join(outargs_rpath, 'mask3d_evaluation.csv')} Text.txt \n"
)
relion_output_nodes_star_file.write(
    f"{os.path.join(outargs_rpath, summary_star_filename)} LogFile.star \n"
)
if plot_filename:
    relion_output_nodes_star_file.write(
        f"{os.path.join(outargs_rpath, plot_filename)} Image.png \n"
    )
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.close()

# Create RELION_JOB_EXIT_SUCCESS file
relion_job_exit_status_file = open(
    os.path.join(outargs_rpath, "RELION_JOB_EXIT_SUCCESS"), "w"
)
relion_job_exit_status_file.close()

print("[GTF_DEBUG] Done")
"""<<< Finishing up"""
