#!/usr/bin/env python

# ***************************************************************************
#
# Copyright (c) 2022-2024 Structural Biology Research Center,
#                         Institute of Materials Structure Science,
#                         High Energy Accelerator Research Organization (KEK)
#
#
# Authors:   Toshio Moriya (toshio.moriya@kek.jp)
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
# This script is to select the best 3D Class for GoToFly on-the-fly system
# It designed to be executed as as an External job type in Relion GUI for AWS GoToCloud Enviroment
# Create: 2022/08/14 Toshio Moriya (KEK, SBRC)
#
# Run with Relion external job (RELION4)
# https://relion.readthedocs.io/en/release-4.0/Reference/Using-RELION.html

# Provide executable in the gui: gtf_relion4_run_select_class3d.py
# Input micrographs.star
# Provide extra parameters in the parameters tab ()
#
# Outputs for RELION
# - selected_data.star
# - selected_model_map.mrc
# ### - summary.star
# - RELION_JOB_EXIT_SUCCESS
# - job_pipeline.star
# Append
# ++++++++++++++++++++++++++++++
# # version 30001
#
# data_pipeline_output_edges
#
# loop_
# _rlnPipeLineEdgeProcess #1
# _rlnPipeLineEdgeToNode #2
# Exteranl/job###/ Exteranl/job###/selected_data.star
# Exteranl/job###/ Exteranl/job###/selected_model_map.mrc
# ++++++++++++++++++++++++++++++

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
from utils import run_subprocess
import asyncio
from config import CONDA_PYTHON_PATH
from deepmasc_core import process_map_files

"""<<< Import"""

"""<<< FUNCTIONS"""

"""USAGE >>>"""
print("This script runs Class3D Selection from RELION Model STAR file in AWS GoToCloud Enviroment")
"""<<< USAGE"""

"""VARIABLES >>>"""
print("running ...")
parser = argparse.ArgumentParser()
# --in_YYY: YYY is the type of the input node: movies, mics, parts, coords, 3dref, or mask,
parser.add_argument("-i", "--input", "--in_parts", type=str, help="RELION requirement! Input particle star file Path (relative)")
parser.add_argument("-o", "--output", type=str, help="RELION requirement! Output job directory path (relative)")
parser.add_argument("-g", "--gpus", type=str, help="GPU ID to use for CryoREAD prediction")
parser.add_argument("--debug", type=bool, help="Enable debug mode to generate full output", default=False)
parser.add_argument("-r", "--reso", choices=["Low", "High"], type=str, help="Resolution to choose the deep learning model, can be Low(>5Å) or High(<5Å)", default="Low")
parser.add_argument("-b", "--batch", type=int, help="Batch size to use for CryoREAD prediction", default=4)

### parser.add_argument("-m", "--model_star",           type=str,                                             help = "Input model star file Path (relative).")
### parser.add_argument("-r", "--script_repo",         type=str,                                              help = "Script repository directory path (full).")
args, unknown = parser.parse_known_args()

inargs_parts = args.input
outargs_rpath = args.output
gpu_ids = args.gpus
# Determine resolution of model to use
reso_input = 8.0 if args.reso == "Low" else 2.0
# batch size to use for CryoREAD prediction
batch_size = args.batch
### model_star_rpath =str( args.model_star)
### script_repo_fpath = str(args.script_repo)
invalid_str = "GTF_INVALID_STR"

print("[GTF_DEBUG] inargs_parts      : %s" % inargs_parts)
print("[GTF_DEBUG] outargs_rpath     : %s" % outargs_rpath)
print("[GTF_DEBUG] gpu_ids          : %s" % gpu_ids)
print("[GTF_DEBUG] batch size       : %s" % batch_size)
print("[GTF_DEBUG] reso input       : %s" % reso_input)
print("[GTF_DEBUG] debug mode       : %s" % args.debug)
print("[GTF_DEBUG] sys.executable   : %s" % sys.executable)
print("[GTF_DEBUG] conda python path : %s" % CONDA_PYTHON_PATH)

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

output_selected_data_star_file_basename = "selected_data.star"
output_selected_map_mrc_file_basename = "selected_model_map.mrc"
print("[GTF_DEBUG] output_selected_data_star_file_basename : %s" % output_selected_data_star_file_basename)
print("[GTF_DEBUG] output_selected_map_mrc_file_basename   : %s" % output_selected_map_mrc_file_basename)

"""<<< VARIABLES"""

"""Preparation >>>"""
# pprint.pprint(sys.path)
# sys.path.append(script_repo_fpath)
# pprint.pprint(sys.path)

assert os.path.exists(inargs_parts), "# Logical Error: Input RELION DATA STAR file must exits."
input_job_dir_rpath, input_data_star_file_basename = os.path.split(inargs_parts)
print("[GTF_DEBUG] input_job_dir_rpath            : %s" % input_job_dir_rpath)
print("[GTF_DEBUG] input_data_star_file_basename  : %s" % input_data_star_file_basename)
"""<<< Preparation"""

"""Selecting the best class >>>"""
# input_data_star_file_basename format : run_it###_data.star
# input_model_star_file_basename format : run_it###_model.star
input_model_star_file_basename = input_data_star_file_basename.replace("data", "model")
input_model_star_rpath = os.path.join(input_job_dir_rpath, input_model_star_file_basename)
print("[GTF_DEBUG] input_model_star_file_basename  : %s" % input_model_star_file_basename)
print("[GTF_DEBUG] input_model_star_rpath          : %s" % input_model_star_rpath)
assert os.path.exists(input_model_star_rpath), "# Logical Error: Input RELION DATA STAR file must exits."

import gtf_relion4_select3d
from gtf_relion4_select3d import is_map_empty

# check if the job is InitialModel or Class3D
if "InitialModel" in input_job_dir_rpath:
    print("[GTF_DEBUG] Running InitialModel Model Selection")
    sort_table = gtf_relion4_select3d.run_initialmodel(input_model_star_rpath, outargs_rpath, relion_project_dir_fpath=None)
elif "Class3D" in input_job_dir_rpath:
    print("[GTF_DEBUG] Running Class3D Model Selection")
    sort_table = gtf_relion4_select3d.run_class3d(input_model_star_rpath, outargs_rpath, relion_project_dir_fpath=None)
else:
    print("[GTF_ERROR] Unknown job type : ", input_job_dir_rpath)
    exit(1)

print("[GTF_DEBUG] sort_table : ", sort_table)

i_sort_table = 0
print("[GTF_DEBUG] Class3D Sort Table Index: Class3D ID, Map File, Resolution, Distribution")
for sort_entry_list in sort_table:
    print(
        "[GTF_DEBUG]   ",
        i_sort_table,
        " : ",
        sort_entry_list[idx_class3d_gtc_class3d_id],
        ", ",
        sort_entry_list[idx_class3d_map_dir_rpath],
        ", ",
        sort_entry_list[idx_class3d_estimated_res],
        ", ",
        sort_entry_list[idx_class3d_distribution],
    )
    i_sort_table += 1
print("")

# Filter out maps with estimated resolution > 30 Å
resolution_threshold = 30.0
filtered_sort_table = []
excluded_maps = []

for sort_entry_list in sort_table:
    estimated_res = float(sort_entry_list[idx_class3d_estimated_res])
    if estimated_res <= resolution_threshold:
        filtered_sort_table.append(sort_entry_list)
    else:
        excluded_maps.append(sort_entry_list)
        print(f"[GTF_DEBUG] Excluding map {sort_entry_list[idx_class3d_map_dir_rpath]} with resolution {estimated_res:.2f} Å (> {resolution_threshold} Å)")

if excluded_maps:
    print(f"[GTF_DEBUG] Excluded {len(excluded_maps)} map(s) due to resolution threshold")
else:
    print(f"[GTF_DEBUG] No maps excluded by resolution threshold")

if not filtered_sort_table:
    print(f"[GTF_ERROR] No maps remain after applying resolution threshold of {resolution_threshold} Å. Exiting.")
    exit(1)

# Update sort_table to use filtered results
sort_table = filtered_sort_table
print(f"[GTF_DEBUG] Proceeding with {len(sort_table)} map(s) after resolution filtering")
print("")

# CryoREAD - using shared core function
FINAL_OUTDIR = os.path.abspath(outargs_rpath)
input_job_dir_rpath_abs = os.path.abspath(input_job_dir_rpath)

# Extract MRC files and class IDs from sort_table
mrc_files = []
class_ids = []
for sort_entry_list in sort_table:
    mrc_file = os.path.join(input_job_dir_rpath_abs, sort_entry_list[idx_class3d_map_dir_rpath].split("/")[-1])
    mrc_file = os.path.abspath(mrc_file)
    class_id = int(sort_entry_list[idx_class3d_gtc_class3d_id])

    mrc_files.append(mrc_file)
    class_ids.append(class_id)

print(f"[GTF_DEBUG] Processing {len(mrc_files)} maps with CryoREAD")

# Process all maps using shared function
# Result format: [(class_id, mrc_file, real_space_cc, cutoff_05), ...]
result_list_cryoREAD = process_map_files(
    mrc_files=mrc_files,
    output_path=FINAL_OUTDIR,
    gpu_ids=gpu_ids,
    batch_size=batch_size,
    reso_input=reso_input,
    debug_mode=args.debug,
    class_ids=class_ids,
)

# Get 1st entry of sorted class3d model table by CryoREAD
first_class3d_sort_entry_list = result_list_cryoREAD[0]
# print('[GTF_DEBUG] first_class3d_sort_entry_list : ', first_class3d_sort_entry_list)

# Print results as table
print("[GTF_DEBUG] CryoREAD Sort Table Index: Class ID, MRC File, Real Space CC, FSC @ 0.5")
i_cryoread_sort_table = 0
for entry in result_list_cryoREAD:
    print(
        "[GTF_DEBUG]   ",
        i_cryoread_sort_table,
        " : ",
        entry[0],
        ", ",  # Class ID
        entry[1],
        ", ",  # MRC File
        entry[2],
        ", ",  # Real Space CC
        entry[3],
    )  # FSC @ 0.5
    i_cryoread_sort_table += 1
print("")

selected_class_id = int(first_class3d_sort_entry_list[0])
print("[GTF_DEBUG] Selected Class ID :", selected_class_id)
print("[GTF_DEBUG] Selected MRC File :", first_class3d_sort_entry_list[1])

"""<<< Selecting the best class"""

"""Copying selected map >>>"""
input_selected_map_file_rpath = first_class3d_sort_entry_list[1]
output_selected_map_file_rpath = os.path.join(outargs_rpath, output_selected_map_mrc_file_basename)
print("[GTF_DEBUG] input_selected_map_file_rpath   : %s" % input_selected_map_file_rpath)
print("[GTF_DEBUG] output_selected_map_file_rpath  : %s" % output_selected_map_file_rpath)
assert os.path.exists(input_selected_map_file_rpath), "# Logical Error: Input RELION MAP MRC file must exits."
shutil.copy2(input_selected_map_file_rpath, output_selected_map_file_rpath)
"""<<< Copying selected map"""

"""Creating selected data star >>>"""
import gtf_relion4_create_select3d_data_star

gtf_relion4_create_select3d_data_star.run(
    inargs_parts, outargs_rpath, selected_class_id, output_selected_data_star_file_basename, relion_project_dir_fpath=None
)
"""<<< Creating slected data star"""

"""Finishing up >>>"""
# See the data_pipeline_nodes table in the default_pipeline.star file of any relion project directory for examples.

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
relion_output_nodes_star_file.write(
    "{} ParticlesData.star.relion \n".format(os.path.join(outargs_rpath, output_selected_data_star_file_basename))
)
relion_output_nodes_star_file.write("{} DensityMap.mrc \n".format(os.path.join(outargs_rpath, output_selected_map_mrc_file_basename)))
# relion_output_nodes_star_file.write(logfile+" 13")
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.close()

relion_job_exit_status_file = open(os.path.join(outargs_rpath, "RELION_JOB_EXIT_SUCCESS"), "w")
relion_job_exit_status_file.close()

print("[GTF_DEBUG] Done")
"""<<< Finishing up"""
