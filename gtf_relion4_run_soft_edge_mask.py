#!/usr/bin/env python

# Author: Han Zhu

from __future__ import print_function

import argparse
import os
import sys

from soft_edge_mask import (
    add_soft_edge_arguments,
    EdtBackendError,
    resolve_soft_edge,
    touches_box,
    voxel_size_of,
    write_soft_mask,
)


def write_summary_star(path, width_px, padding_px, width_a, padding_a, voxel, ini_threshold, touches):
    with open(path, "w") as f:
        f.write("\n")
        f.write("# version 30001\n")
        f.write("data_soft_edge_mask\n")
        f.write("\n")
        f.write("loop_\n")
        f.write("_rlnSoftEdgeWidthPixel #1\n")
        f.write("_rlnSoftEdgePaddingPixel #2\n")
        f.write("_rlnSoftEdgeWidthAngstrom #3\n")
        f.write("_rlnSoftEdgePaddingAngstrom #4\n")
        f.write("_rlnSoftEdgeVoxelSize #5\n")
        f.write("_rlnSoftEdgeIniThreshold #6\n")
        f.write("_rlnSoftEdgeTouchesBox #7\n")
        f.write(
            f"{width_px:.6f} {padding_px:.6f} {width_a:.6f} {padding_a:.6f} "
            f"{voxel:.6f} {ini_threshold:.6f} {int(touches)}\n"
        )
        f.write("\n")


def run(parser, args):
    inargs_mask = os.path.abspath(args.input)
    outargs_rpath = os.path.abspath(args.output)

    print("[GTF_DEBUG] inargs_mask         : %s" % inargs_mask)
    print("[GTF_DEBUG] outargs_rpath       : %s" % outargs_rpath)
    print("[GTF_DEBUG] width_px            : %s" % args.width_px)
    print("[GTF_DEBUG] width_A             : %s" % args.width_A)
    print("[GTF_DEBUG] extend_px           : %s" % args.extend_px)
    print("[GTF_DEBUG] extend_A            : %s" % args.extend_A)
    print("[GTF_DEBUG] ini_threshold       : %s" % args.ini_threshold)
    print("[GTF_DEBUG] preset              : %s" % args.preset)
    print("[GTF_DEBUG] resolution          : %s" % args.resolution)
    print("[GTF_DEBUG] edt_backend         : %s" % args.edt_backend)
    print("[GTF_DEBUG] taichi_arch         : %s" % args.taichi_arch)

    assert os.path.exists(inargs_mask), (
        f"# Logical Error: Input mask file ({inargs_mask}) must exist."
    )

    voxel = voxel_size_of(parser, inargs_mask)
    width_px, padding_px, width_a, padding_a = resolve_soft_edge(parser, args, voxel)

    print("[GTF_DEBUG] Creating the soft mask...")
    output_mask = os.path.join(outargs_rpath, "soft_mask.mrc")
    try:
        soft = write_soft_mask(
            inargs_mask, output_mask, width_px, extend=padding_px, ini_threshold=args.ini_threshold,
            edt_backend=args.edt_backend, taichi_arch=args.taichi_arch,
        )
    except EdtBackendError as e:
        parser.error(str(e))
    touches = touches_box(soft)
    print(
        "[GTF_DEBUG] width %g px (%g A), padding %g px (%g A), voxel size %g A, touches box face: %s"
        % (width_px, width_a, padding_px, padding_a, voxel, "yes" if touches else "no")
    )
    if touches:
        print("[GTF_DEBUG] WARNING: the soft mask is non-zero on a box face; the soft edge is clipped by the box.")

    summary_star_path = os.path.join(outargs_rpath, "soft_edge_mask_summary.star")
    write_summary_star(
        summary_star_path, width_px, padding_px, width_a, padding_a, voxel, args.ini_threshold, touches
    )
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
        f.write(f"{output_mask} Mask3D.mrc \n")
        f.write(f"{summary_star_path} LogFile.star \n")
        f.write("\n")

    with open(os.path.join(outargs_rpath, "RELION_JOB_EXIT_SUCCESS"), "w") as f:
        pass


if __name__ == "__main__":
    print("[GTF_DEBUG] Full command:", " ".join(sys.argv))

    print("This script makes a soft mask from a binary mask (equal to relion_mask_create)")

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
    add_soft_edge_arguments(parser)

    args, unknown = parser.parse_known_args()
    os.makedirs(os.path.abspath(args.output), exist_ok=True)

    try:
        run(parser, args)
    except SystemExit as e:
        # parser.error() inside run() exits with code 2
        with open(os.path.join(os.path.abspath(args.output), "RELION_JOB_EXIT_FAILURE"), "w"):
            pass
        sys.exit(e.code if e.code else 1)
    except Exception:
        with open(os.path.join(os.path.abspath(args.output), "RELION_JOB_EXIT_FAILURE"), "w"):
            pass
        raise

    print("[GTF_DEBUG] Done")
