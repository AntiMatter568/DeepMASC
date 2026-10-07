#!/usr/bin/env python

# Author: Han Zhu

from __future__ import print_function

import argparse
import os
import shutil
import sys
from pathlib import Path

from eval_refinement_mask import add_criterion_arguments
from soft_edge_mask import add_edt_backend_arguments
from optimal_soft_edge_relion import run_soft_edge_search

if __name__ == "__main__":
    print("[GTF_DEBUG] Full command:", " ".join(sys.argv))

    print("This script searches the narrowest soft edge width at which a 3D mask passes the PRFSC criteria")

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
        type=float,
        default=0,
        help="Mask extension in pixels before the soft edge (default 0)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["distance", "relion"],
        default="distance",
        help="Soft mask engine: distance transform (default, same values as relion_mask_create) or relion_mask_create",
    )
    parser.add_argument(
        "--relion_bin",
        type=str,
        default=None,
        help="Directory with the RELION binaries (default: found on PATH)",
    )
    parser.add_argument(
        "-j",
        "--n_threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of threads for relion_mask_create (engine relion)",
    )

    add_edt_backend_arguments(parser)
    add_criterion_arguments(parser)

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
    print("[GTF_DEBUG] engine              : %s" % args.engine)
    print("[GTF_DEBUG] relion_bin          : %s" % args.relion_bin)
    print("[GTF_DEBUG] n_threads           : %s" % args.n_threads)
    print("[GTF_DEBUG] edt_backend         : %s" % args.edt_backend)
    print("[GTF_DEBUG] taichi_arch         : %s" % args.taichi_arch)
    print("[GTF_DEBUG] reference_fsc       : %s" % args.reference_fsc)
    print("[GTF_DEBUG] reference_threshold : %s" % args.reference_threshold)
    print("[GTF_DEBUG] margin_shells       : %s" % args.margin_shells)

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

    print("[GTF_DEBUG] Starting optimal soft edge parameter search...")

    optimal_summary = run_soft_edge_search(
        input_map_path=inargs_mask,
        output_folder=outargs_rpath,
        half_map=half_map,
        extend_inimask=args.extend_inimask,
        engine=args.engine,
        n_threads=args.n_threads,
        relion_bin=args.relion_bin,
        edt_backend=args.edt_backend,
        taichi_arch=args.taichi_arch,
        reference_fsc=args.reference_fsc,
        reference_threshold=args.reference_threshold,
        margin_shells=args.margin_shells,
    )

    print("[GTF_DEBUG] Optimal soft edge found:")
    print("[GTF_DEBUG]   Soft Edge Width: %d" % optimal_summary["optimal_soft_edge_width"])
    print("[GTF_DEBUG]   PRFSC pass: %s" % optimal_summary["prfsc_pass"])
    print("[GTF_DEBUG]   Stop reason: %s" % optimal_summary["stop_reason"])

    emdid = Path(inargs_mask).stem
    optimal_mask_mrc = optimal_summary["optimal_mask_path"]

    output_optimal_mask = os.path.join(outargs_rpath, "optimal_mask.mrc")
    shutil.copy(optimal_mask_mrc, output_optimal_mask)
    print("[GTF_DEBUG] Optimal mask copied to: %s" % output_optimal_mask)

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
        f.write("_rlnOptimalSoftEdgeWidth #1\n")
        f.write("_rlnPrfscPass #2\n")
        f.write("_rlnNoPassingWidth #3\n")
        f.write("_rlnSoftEdgeStopReason #4\n")
        f.write("_rlnOptimalExtendInimask #5\n")
        f.write("_rlnOptimalMaskedResolution0143 #6\n")
        f.write("_rlnOptimalPhaseRandZeroResolution #7\n")
        f.write("_rlnOptimalMaskedZeroResolution #8\n")
        f.write("_rlnOptimalCorrectedResolution #9\n")
        f.write("_rlnWidthsTried #10\n")
        f.write("_rlnPrfscReferenceFsc #11\n")
        f.write("_rlnPrfscReferenceThreshold #12\n")
        f.write("_rlnPrfscMarginShells #13\n")
        f.write(
            f"{optimal_summary['optimal_soft_edge_width']} "
            f"{int(optimal_summary['prfsc_pass'])} "
            f"{int(optimal_summary['no_passing_width'])} "
            f"{optimal_summary['stop_reason']} "
            f"{optimal_summary['optimal_extend_inimask']} "
            f"{_fmt(optimal_summary['optimal_masked_res_0_143'])} "
            f"{_fmt(optimal_summary['optimal_phase_rand_zero_res'])} "
            f"{_fmt(optimal_summary['optimal_masked_zero_res'])} "
            f"{_fmt(optimal_summary['optimal_corrected_resolution'])} "
            f"{len(optimal_summary['widths_tried'])} "
            f"{optimal_summary['reference_fsc']} "
            f"{optimal_summary['reference_threshold']} "
            f"{optimal_summary['margin_shells']}\n"
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
