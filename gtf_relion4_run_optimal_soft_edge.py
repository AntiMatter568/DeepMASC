#!/usr/bin/env python

# Author: Han Zhu

from __future__ import print_function

import argparse
import os
import shutil
import sys
from pathlib import Path

from optimal_soft_edge_relion import run_soft_edge_grid_search

if __name__ == "__main__":
    print("[GTF_DEBUG] Full command:", " ".join(sys.argv))

    print("This script runs optimal soft edge parameter search for 3D masks")

    print("running ...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        "--in_mask",
        type=str,
        help="RELION requirement! Input binary mask MRC file path (relative)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="RELION requirement! Output job directory path (relative)",
    )
    parser.add_argument(
        "--half_map",
        type=str,
        required=True,
        help="Path to half map 1 (half map 2 auto-detected)",
    )
    parser.add_argument(
        "-e",
        "--extend_inimask",
        type=str,
        default="0,2,3,4",
        help="Extend initial mask values (comma-separated, e.g., '0,2,3,4')",
    )
    parser.add_argument(
        "-s",
        "--soft_edge_widths",
        type=str,
        default="5,10,15,20,25",
        help="Soft edge width values (comma-separated, e.g., '5,10,15,20,25')",
    )
    parser.add_argument(
        "-j",
        "--n_threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of threads for relion_mask_create",
    )

    args, unknown = parser.parse_known_args()

    inargs_mask = args.input
    outargs_rpath = args.output
    inargs_mask = os.path.abspath(inargs_mask)
    outargs_rpath = os.path.abspath(outargs_rpath)
    half_map = os.path.abspath(args.half_map)

    print("[GTF_DEBUG] inargs_mask         : %s" % inargs_mask)
    print("[GTF_DEBUG] outargs_rpath       : %s" % outargs_rpath)
    print("[GTF_DEBUG] half_map            : %s" % half_map)
    print("[GTF_DEBUG] extend_inimask      : %s" % args.extend_inimask)
    print("[GTF_DEBUG] soft_edge_widths    : %s" % args.soft_edge_widths)
    print("[GTF_DEBUG] n_threads           : %s" % args.n_threads)

    assert os.path.exists(inargs_mask), (
        f"# Logical Error: Input mask file ({inargs_mask}) must exist."
    )
    assert os.path.exists(half_map), (
        f"# Logical Error: Half map file ({half_map}) must exist."
    )

    half_map_2 = half_map.replace("_half1", "_half2").replace("_halfA", "_halfB")
    assert os.path.exists(half_map_2), (
        f"# Logical Error: Half map 2 file ({half_map_2}) must exist."
    )

    os.makedirs(outargs_rpath, exist_ok=True)

    try:
        extend_inimask_values = [int(x.strip()) for x in args.extend_inimask.split(",")]
    except ValueError:
        raise ValueError(
            f"# Logical Error: Invalid extend_inimask values: {args.extend_inimask}"
        )

    try:
        soft_edge_widths = [int(x.strip()) for x in args.soft_edge_widths.split(",")]
    except ValueError:
        raise ValueError(
            f"# Logical Error: Invalid soft_edge_widths values: {args.soft_edge_widths}"
        )

    print("[GTF_DEBUG] Starting optimal soft edge parameter search...")

    optimal_summary = run_soft_edge_grid_search(
        input_map_path=inargs_mask,
        output_folder=outargs_rpath,
        half_map=half_map,
        extend_inimask_values=extend_inimask_values,
        soft_edge_widths=soft_edge_widths,
        n_threads=args.n_threads,
    )

    if optimal_summary is None:
        raise ValueError(
            "# Logical Error: No valid results found during soft edge parameter search"
        )

    print("[GTF_DEBUG] Optimal parameters found:")
    print(
        "[GTF_DEBUG]   Extend Inimask: %d" % optimal_summary["optimal_extend_inimask"]
    )
    print(
        "[GTF_DEBUG]   Soft Edge Width: %d" % optimal_summary["optimal_soft_edge_width"]
    )
    print(
        "[GTF_DEBUG]   Corrected Resolution: %.3f Å"
        % optimal_summary["optimal_corrected_resolution"]
    )

    emdid = Path(inargs_mask).stem
    opt_extend = optimal_summary["optimal_extend_inimask"]
    opt_soft_edge = optimal_summary["optimal_soft_edge_width"]

    optimal_mask_mrc = os.path.join(
        outargs_rpath,
        f"extend_{opt_extend}_soft_edge_{opt_soft_edge}",
        f"{emdid}_extend_{opt_extend}_soft_edge_{opt_soft_edge}.mrc",
    )

    output_optimal_mask = os.path.join(outargs_rpath, "optimal_mask.mrc")
    if os.path.exists(optimal_mask_mrc):
        shutil.copy(optimal_mask_mrc, output_optimal_mask)
        print("[GTF_DEBUG] Optimal mask copied to: %s" % output_optimal_mask)
    else:
        print(
            "[GTF_DEBUG] Warning: Optimal mask file not found at: %s" % optimal_mask_mrc
        )

    import math

    print("[GTF_DEBUG] Creating summary star file...")
    summary_star_path = os.path.join(outargs_rpath, "optimal_soft_edge_summary.star")

    def _fmt(val, decimals=6):
        if isinstance(val, float) and math.isnan(val):
            return "-1.000000"
        return f"{val:.{decimals}f}"

    with open(summary_star_path, "w") as f:
        f.write("\n")
        f.write("# version 30001\n")
        f.write("data_optimal_soft_edge\n")
        f.write("\n")
        f.write("loop_\n")
        f.write("_rlnOptimalExtendInimask #1\n")
        f.write("_rlnOptimalSoftEdgeWidth #2\n")
        f.write("_rlnOptimalCorrectedResolution #3\n")
        f.write("_rlnOptimalUnmaskedResolution0143 #4\n")
        f.write("_rlnOptimalMaskedResolution0143 #5\n")
        f.write("_rlnOptimalCorrectionMagnitude #6\n")
        f.write("_rlnOptimalPhaseRandNoiseFloor #7\n")
        f.write("_rlnCriterionMet #8\n")
        f.write("_rlnTotalCombinationsTested #9\n")
        f.write("_rlnValidResultsCount #10\n")
        f.write(
            f"{opt_extend} {opt_soft_edge} "
            f"{_fmt(optimal_summary['optimal_corrected_resolution'])} "
            f"{_fmt(optimal_summary.get('optimal_unmasked_res_0_143', float('nan')))} "
            f"{_fmt(optimal_summary.get('optimal_masked_res_0_143', float('nan')))} "
            f"{_fmt(optimal_summary.get('optimal_correction_magnitude', float('nan')))} "
            f"{_fmt(optimal_summary.get('optimal_phase_rand_noise_floor', float('nan')))} "
            f"{int(optimal_summary['criterion_met'])} "
            f"{optimal_summary['total_results']} "
            f"{optimal_summary['valid_results']}\n"
        )
        f.write("\n")
    print("[GTF_DEBUG] Summary star file saved: %s" % summary_star_path)

    print("[GTF_DEBUG] Creating RELION output files...")

    with open(os.path.join(outargs_rpath, "RELION_OUTPUT_NODES.star"), "w") as f:
        f.write("\n")
        f.write("# version 30001\n")
        f.write("data_output_nodes\n")
        f.write("\n")
        f.write("loop_\n")
        f.write("_rlnPipeLineNodeName #1 \n")
        f.write("_rlnPipeLineNodeTypeLabel #2 \n")
        if os.path.exists(output_optimal_mask):
            f.write(f"{output_optimal_mask} DensityMap.mrc \n")
        f.write(f"{summary_star_path} LogFile.star \n")
        combined_csv = os.path.join(outargs_rpath, f"{emdid}_all_parameter_results.csv")
        if os.path.exists(combined_csv):
            f.write(f"{combined_csv} Text.txt \n")
        f.write("\n")

    with open(os.path.join(outargs_rpath, "RELION_JOB_EXIT_SUCCESS"), "w") as f:
        pass

    print("[GTF_DEBUG] Done")
