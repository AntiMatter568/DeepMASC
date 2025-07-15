#!/usr/bin/env python

# Author: Han Zhu

from __future__ import print_function

"""Import >>>"""
import argparse
import os
import shutil  # copyfile
import tempfile
import select
import sys
import pprint
from pathlib import Path
from glob import glob
from utils import run_subprocess
import asyncio

if __name__ == "__main__":

    print("[GTF_DEBUG] Full command:", " ".join(sys.argv))

    """<<< Import"""

    """USAGE >>>"""
    print("This script runs Auto Contour Level Determination from MRC file")
    """<<< USAGE"""

    """VARIABLES >>>"""
    print("running ...")
    parser = argparse.ArgumentParser()
    # --in_YYY: YYY is the type of the input node: movies, mics, parts, coords, 3dref, or mask,
    parser.add_argument("-i", "--input", "--in_3dref", type=str, help="RELION requirement! Input mrc file Path (relative)")
    parser.add_argument("-o", "--output", type=str, help="RELION requirement! Output job directory path (relative)")
    parser.add_argument("-g", "--gpus", type=str, help="GPU ID to use for CryoREAD prediction", default="0")
    parser.add_argument("-b", "--batch", type=int, help="Batch size to use for CryoREAD prediction", default=4)
    parser.add_argument("-n", "--num_components", type=int, default=2, help="Number of components for mixture model")
    parser.add_argument("-r", "--refinement_mask", type=bool, help="Generate more fine-grained mask for refinement", default=False)
    parser.add_argument("-m", "--morph_radius", type=int, default=3, help="The radius for morphological operations (opening, closing)")
    parser.add_argument(
        "-d",
        "--mask_diameter",
        type=int,
        default=95,
        choices=range(0, 101),
        help="The diameter of the mask in percentage to the shortest dimension of the map (from 0 to 100), set to 0 to disable",
    )
    parser.add_argument("-a", "--aggressive", type=bool, help="Use more aggressive mask cutoff when using GMM mask", default=False)
    parser.add_argument("--debug", type=bool, help="Debug mode", default=False)

    args, unknown = parser.parse_known_args()

    inargs_parts = args.input
    outargs_rpath = args.output
    inargs_parts = os.path.abspath(inargs_parts)
    outargs_rpath = os.path.abspath(outargs_rpath)
    gpu_ids = args.gpus
    # batch size to use for CryoREAD prediction
    batch_size = args.batch
    invalid_str = "GTF_INVALID_STR"

    print("[GTF_DEBUG] inargs_parts      : %s" % inargs_parts)
    print("[GTF_DEBUG] outargs_rpath     : %s" % outargs_rpath)
    print("[GTF_DEBUG] gpu_ids          : %s" % gpu_ids)
    ## print('[GTF_DEBUG] model_star_rpath  : %s' % model_star_rpath)
    ### print('[GTF_DEBUG] script_repo_fpath : %s' % script_repo_fpath)

    # Define constants
    ### cryolo_predict_exe = 'cryolo_predict.py'
    ### print('[GTF_DEBUG] cryolo_predict_exe             : %s' % cryolo_predict_exe)

    # For Class3D Model Classes (class3d) parameters file format as defined in gtf_relion4_select3d
    i_enum = -1
    i_enum += 1
    idx_class3d_map_dir_rpath = i_enum
    i_enum += 1
    idx_class3d_distribution = i_enum
    i_enum += 1
    idx_class3d_accuracy_rot = i_enum
    i_enum += 1
    idx_class3d_accuracy_shift = i_enum
    i_enum += 1
    idx_class3d_estimated_res = i_enum
    i_enum += 1
    idx_class3d_completeness = i_enum
    i_enum += 1
    idx_class3d_gtc_class3d_id = i_enum
    i_enum += 1
    n_idx_class3d = i_enum

    """<<< VARIABLES"""

    """Preparation >>>"""
    # pprint.pprint(sys.path)
    # sys.path.append(script_repo_fpath)
    # pprint.pprint(sys.path)

    assert os.path.exists(inargs_parts), f"# Logical Error: Input MRC file ({inargs_parts}) must exits."
    input_job_dir_rpath, input_data_mrc_file_basename = os.path.split(inargs_parts)
    print("[GTF_DEBUG] input_job_dir_rpath            : %s" % input_job_dir_rpath)
    print("[GTF_DEBUG] input_data_mrc_file_basename  : %s" % input_data_mrc_file_basename)
    """<<< Preparation"""

    """Input >>>"""

    # input_selected_map_mrc_file = os.path.join(input_job_dir_rpath, input_data_mrc_file_basename)
    # print("[GTF_DEBUG] input_selected_map_mrc_file_basename   : %s" % input_selected_map_mrc_file)
    # assert os.path.exists(input_selected_map_mrc_file), "# Logical Error: Input Select Model Map MRC file must exist."

    CURR_SCRIPT_PATH = Path(__file__).absolute().parent
    TEMP_CURR_DIR = os.getcwd()
    os.chdir(CURR_SCRIPT_PATH)

    """AutoContour >>>"""
    cmd = [
        "pixi",
        "run",
        "autocontour",
        f"--input_map_path={inargs_parts}",
        f"--output_folder={outargs_rpath}",
        f"--gpu={gpu_ids}",
        f"--batch_size={batch_size}",
        f"--num_components={args.num_components}",
        f"--morph_radius={args.morph_radius}",
        f"--mask_diameter={args.mask_diameter}",
    ]

    if args.refinement_mask:
        cmd.append("--refinement_mask")
    if args.aggressive:
        cmd.append("--aggressive")

    print("[GTF_DEBUG] AutoContour Command : ", " ".join(cmd))

    exit_code = asyncio.run(run_subprocess(cmd))
    if exit_code != 0:
        raise ValueError(f"# Logical Error: AutoContour failed with exit code {exit_code}")

    os.chdir(TEMP_CURR_DIR)
    """<<< AutoContour"""

    """Finishing up >>>"""
    # See the data_pipeline_nodes table in the default_pipeline.star file of any relion project directory for examples.

    output_mask_mrc_file = os.path.join(outargs_rpath, "prot_mask_final.mrc")
    assert os.path.exists(output_mask_mrc_file), f"# Logical Error: Output Mask MRC file ({output_mask_mrc_file}) must exist."

    print("Creating RELION_OUTPUT_NODES star file ...")
    # relion_output_nodes_star_file = open(os.path.join(outargs_rpath, "RELION_OUTPUT_NODES.star"),"w+")
    relion_output_nodes_star_file = open(os.path.join(outargs_rpath, "RELION_OUTPUT_NODES.star"), "w")
    relion_output_nodes_star_file.write("\n")
    relion_output_nodes_star_file.write("# version 30001\n")
    relion_output_nodes_star_file.write("data_output_nodes\n")
    relion_output_nodes_star_file.write("\n")
    relion_output_nodes_star_file.write("loop_\n")
    relion_output_nodes_star_file.write("_rlnPipeLineNodeName #1 \n")
    # relion_output_nodes_star_file.write("_rlnPipeLineNodeType #2\n")
    relion_output_nodes_star_file.write("_rlnPipeLineNodeTypeLabel #2 \n")
    # relion_output_nodes_star_file.write(
    #     "{} ParticlesData.star.relion \n".format(os.path.join(outargs_rpath, output_selected_data_star_file_basename))
    # )
    relion_output_nodes_star_file.write("{} DensityMap.mrc \n".format(os.path.join(outargs_rpath, output_mask_mrc_file)))
    # relion_output_nodes_star_file.write(logfile+" 13")
    relion_output_nodes_star_file.write("\n")
    relion_output_nodes_star_file.close()

    relion_job_exit_status_file = open(os.path.join(outargs_rpath, "RELION_JOB_EXIT_SUCCESS"), "w")
    relion_job_exit_status_file.close()

    output_contour_level_file = glob(os.path.join(outargs_rpath, "*revised_contour.txt"))
    if len(output_contour_level_file) != 1:
        raise ValueError(f"# Logical Error: Output Contour Level file does not exist.")
    output_contour_level_file = output_contour_level_file[0]

    with open(output_contour_level_file, "r") as f:
        lines = f.read().splitlines()
        print("[GTF_DEBUG] Contour Level File: ", lines)
        contour_conservative = float(lines[0].split()[-1])
        contour_aggressive = float(lines[1].split()[-1])
        masked_percentage = float(lines[2].split()[-1])

    print("Creating Contour Level star file ...")
    relion_contour_level_star_file = open(os.path.join(outargs_rpath, "CONTOUR_LEVEL.star"), "w")
    relion_contour_level_star_file.write("\n")
    relion_contour_level_star_file.write("# version 30001\n")
    relion_contour_level_star_file.write("data_general\n")
    relion_contour_level_star_file.write("\n")
    relion_contour_level_star_file.write(f"_rlnContourLevelConservative #1 {contour_conservative}\n")
    relion_contour_level_star_file.write(f"_rlnContourLevelAggressive #2 {contour_aggressive}\n")
    relion_contour_level_star_file.write(f"_rlnMaskedPercentage #3 {masked_percentage}\n")
    relion_contour_level_star_file.write("\n")
    relion_contour_level_star_file.close()

    print("[GTF_DEBUG] Done")
    """<<< Finishing up"""
