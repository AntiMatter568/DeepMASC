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

import subprocess

"""Import >>>"""
import argparse
import os
import shutil  # copyfile
import tempfile
import sys
import pprint
from pathlib import Path

"""<<< Import"""

"""USAGE >>>"""
print("This script runs Class3D Selection from RELION Model STAR file in AWS GoToCloud Enviroment")
"""<<< USAGE"""

"""VARIABLES >>>"""
print('running ...')
parser = argparse.ArgumentParser()
# --in_YYY: YYY is the type of the input node: movies, mics, parts, coords, 3dref, or mask,
parser.add_argument("-i", "--input", "--in_parts", type=str,
                    help="RELION requirement! Input particle star file Path (relative)")
parser.add_argument("-o", "--output", type=str, help="RELION requirement! Output job directory path (relative)")
parser.add_argument("-g", "--device", type=str, help="GPU ID to use for CryoREAD prediction")
parser.add_argument("--temp", type=str, help="Temporary directory path", default="/tmp")
parser.add_argument("--debug", type=bool, help="Enable debug mode to generate full output", default=False)
parser.add_argument("-r","--reso", type=str, help="Resolution to choose the deep learning model", default="Low")

### parser.add_argument("-m", "--model_star",           type=str,                                             help = "Input model star file Path (relative).")
### parser.add_argument("-r", "--script_repo",         type=str,                                              help = "Script repository directory path (full).")
args, unknown = parser.parse_known_args()

inargs_parts = args.input
outargs_rpath = args.output
gpu_ids = args.device
### model_star_rpath =str( args.model_star)
### script_repo_fpath = str(args.script_repo)
invalid_str = "GTF_INVALID_STR"

print('[GTF_DEBUG] inargs_parts      : %s' % inargs_parts)
print('[GTF_DEBUG] outargs_rpath     : %s' % outargs_rpath)
print('[GTF_DEBUG] gpu_ids          : %s' % gpu_ids)
## print('[GTF_DEBUG] model_star_rpath  : %s' % model_star_rpath)
### print('[GTF_DEBUG] script_repo_fpath : %s' % script_repo_fpath)

# Define constants
### cryolo_predict_exe = 'cryolo_predict.py'
### print('[GTF_DEBUG] cryolo_predict_exe             : %s' % cryolo_predict_exe)

# For Class3D Model Classes (class3d) parameters file format as defined in gtf_relion4_select3d
i_enum = -1
i_enum += 1;
idx_class3d_map_dir_rpath = i_enum
i_enum += 1;
idx_class3d_distribution = i_enum
i_enum += 1;
idx_class3d_accuracy_rot = i_enum
i_enum += 1;
idx_class3d_accuracy_shift = i_enum
i_enum += 1;
idx_class3d_estimated_res = i_enum
i_enum += 1;
idx_class3d_completeness = i_enum
i_enum += 1;
idx_class3d_gtc_class3d_id = i_enum
i_enum += 1;
n_idx_class3d = i_enum

output_selected_data_star_file_basename = "selected_data.star"
output_selected_map_mrc_file_basename = "selected_model_map.mrc"
print('[GTF_DEBUG] output_selected_data_star_file_basename : %s' % output_selected_data_star_file_basename)
print('[GTF_DEBUG] output_selected_map_mrc_file_basename   : %s' % output_selected_map_mrc_file_basename)

"""<<< VARIABLES"""

"""Preparation >>>"""
# pprint.pprint(sys.path)
# sys.path.append(script_repo_fpath)
# pprint.pprint(sys.path)

assert os.path.exists(inargs_parts), '# Logical Error: Input RELION DATA STAR file must exits.'
input_job_dir_rpath, input_data_star_file_basename = os.path.split(inargs_parts)
print('[GTF_DEBUG] input_job_dir_rpath            : %s' % input_job_dir_rpath)
print('[GTF_DEBUG] input_data_star_file_basename  : %s' % input_data_star_file_basename)
"""<<< Preparation"""

"""Selecting the best class >>>"""
# input_data_star_file_basename format : run_it###_data.star
# input_model_star_file_basename format : run_it###_model.star
input_model_star_file_basename = input_data_star_file_basename.replace('data', 'model')
input_model_star_rpath = os.path.join(input_job_dir_rpath, input_model_star_file_basename)
print('[GTF_DEBUG] input_model_star_file_basename  : %s' % input_model_star_file_basename)
print('[GTF_DEBUG] input_model_star_rpath          : %s' % input_model_star_rpath)
assert os.path.exists(input_model_star_rpath), '# Logical Error: Input RELION DATA STAR file must exits.'

class3d_sort_table = []
import gtf_relion4_select3d

class3d_sort_table = gtf_relion4_select3d.run(input_model_star_rpath, outargs_rpath, relion_project_dir_fpath=None)
i_class3d_sort_table = 0
print('[GTF_DEBUG] Class3D Sort Table Index: Class3D ID, Map File, Resolution, Distribution')
for class3d_sort_entry_list in class3d_sort_table:
    print('[GTF_DEBUG]   ', i_class3d_sort_table, ' : ', class3d_sort_entry_list[idx_class3d_gtc_class3d_id], ', ',
          class3d_sort_entry_list[idx_class3d_map_dir_rpath], ', ', class3d_sort_entry_list[idx_class3d_estimated_res],
          ', ', class3d_sort_entry_list[idx_class3d_distribution])
    i_class3d_sort_table += 1
print('')

# CryoREAD
CURR_SCIPT_PATH = Path(__file__).absolute().parent
CRYOREAD_PATH = CURR_SCIPT_PATH / "CryoREAD" / "main.py"
OUTDIR = os.path.abspath(outargs_rpath)
TEMP_CURR_DIR = os.getcwd()
input_job_dir_rpath_abs = os.path.abspath(input_job_dir_rpath)
os.chdir(CURR_SCIPT_PATH)
result_list_cryoREAD = []

# Create the temp directory if it doesn't exist
os.makedirs(args.temp, exist_ok=True)

temp_dir = tempfile.TemporaryDirectory(dir=args.temp)
try:
    temp_dir_name = temp_dir.name
    print('[GTF_DEBUG] Created temporary directory', temp_dir_name)

    for class3d_sort_entry_list in class3d_sort_table:
        mrc_file = os.path.join(input_job_dir_rpath_abs,
                                class3d_sort_entry_list[idx_class3d_map_dir_rpath].split("/")[-1])
        print('[GTF_DEBUG] Running CryoREAD on mrc_file : ', mrc_file)
        map_name = Path(mrc_file).stem.split(".")[0].strip()
        curr_out_dir = os.path.join(temp_dir.name, map_name)
        print('[GTF_DEBUG] curr_out_dir : ', curr_out_dir)
        os.makedirs(curr_out_dir, exist_ok=True)
        class_id = int(class3d_sort_entry_list[idx_class3d_gtc_class3d_id])
        seg_map_path = os.path.join(curr_out_dir, "input_segment.mrc")
        prot_prob_path = os.path.join(curr_out_dir, "mask_protein.mrc")
        cmd = [
            "pixi",
            "run",
            "cryoread",
            "--mode=0",
            f"-F={mrc_file}",
            "--contour=0",
            f"--gpu={gpu_ids}",
            f"--batch_size=8",  # TODO: Automatic Batch Sizing or expose this as parameter
            f"--prediction_only",
            f"--resolution=8.0",
            f"--output={curr_out_dir}",
        ]
        print('[GTF_DEBUG] cmd : ', " ".join(cmd))
        process = subprocess.run(cmd, shell=False, text=True)
        output_file = os.path.join(curr_out_dir, "CCC_FSC05.txt")
        metrics = []
        with open(output_file, "r") as f:
            metrics = f.read().splitlines()

        real_space_cc = float(metrics[0])
        cutoff_05 = float(metrics[1])

        if args.debug:
            # copy all files to final output dir
            shutil.copytree(curr_out_dir, os.path.join(OUTDIR, map_name))
        else:
            # copyfiles to final output dir
            shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_base_prob.mrc"),
                        os.path.join(OUTDIR, f"{map_name}_chain_base_prob.mrc"))
            shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_phosphate_prob.mrc"),
                        os.path.join(OUTDIR, f"{map_name}_chain_phosphate_prob.mrc"))
            shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_sugar_prob.mrc"),
                        os.path.join(OUTDIR, f"{map_name}_chain_sugar_prob.mrc"))
            shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_protein_prob.mrc"),
                        os.path.join(OUTDIR, f"{map_name}_chain_protein_prob.mrc"))
            shutil.copy(seg_map_path, os.path.join(OUTDIR, f"{map_name}_segment.mrc"))
            shutil.copy(prot_prob_path, os.path.join(OUTDIR, f"{map_name}_mask_protein.mrc"))
            shutil.copy(output_file, os.path.join(OUTDIR, f"{map_name}_CCC_FSC05.txt"))

        # try:
        #     real_space_cc = calc_map_ccc(seg_map_path, prot_prob_path)[0]
        # except:
        #     print('[GTF_DEBUG] CCC calculation failed on : ', mrc_file)
        #     real_space_cc = 0.0
        #
        # try:
        #     x, fsc, cutoff_05, cutoff_0143 = calculate_fsc(seg_map_path, prot_prob_path)
        # except:
        #     print('[GTF_DEBUG] FSC calculation failed on : ', mrc_file)
        #     cutoff_05 = 0.0

        result_list_cryoREAD.append([class_id, mrc_file, real_space_cc, cutoff_05])
        result_list_cryoREAD.sort(key=lambda elem: elem[2], reverse=True)
except:
    print('[GTF_DEBUG] Error running CryoREAD')

temp_dir.cleanup()  # delete temp dir
os.chdir(TEMP_CURR_DIR)

# Get 1st entry of sorted class3d model table by CryoREAD
first_class3d_sort_entry_list = result_list_cryoREAD[0]
# print('[GTF_DEBUG] first_class3d_sort_entry_list : ', first_class3d_sort_entry_list)

# Print results as table
print('[GTF_DEBUG] CryoREAD Sort Table Index: Class ID, MRC File, Real Space CC, FSC @ 0.5')
i_cryoread_sort_table = 0
for entry in result_list_cryoREAD:
    print('[GTF_DEBUG]   ', i_cryoread_sort_table, ' : ',
          entry[0], ', ',  # Class ID
          entry[1], ', ',  # MRC File
          entry[2], ', ',  # Real Space CC
          entry[3])  # FSC @ 0.5
    i_cryoread_sort_table += 1
print('')

selected_class_id = int(first_class3d_sort_entry_list[0])
print('[GTF_DEBUG] Selected Class ID :', selected_class_id)
print('[GTF_DEBUG] Selected MRC File :', first_class3d_sort_entry_list[1])

"""<<< Selecting the best class"""

"""Copying selected map >>>"""
input_selected_map_file_rpath = first_class3d_sort_entry_list[1]
output_selected_map_file_rpath = os.path.join(outargs_rpath, output_selected_map_mrc_file_basename)
print('[GTF_DEBUG] input_selected_map_file_rpath   : %s' % input_selected_map_file_rpath)
print('[GTF_DEBUG] output_selected_map_file_rpath  : %s' % output_selected_map_file_rpath)
assert os.path.exists(input_selected_map_file_rpath), '# Logical Error: Input RELION MAP MRC file must exits.'
shutil.copy2(input_selected_map_file_rpath, output_selected_map_file_rpath)
"""<<< Copying selected map"""

"""Creating selected data star >>>"""
import gtf_relion4_create_select3d_data_star

gtf_relion4_create_select3d_data_star.run(inargs_parts, outargs_rpath, selected_class_id,
                                          output_selected_data_star_file_basename, relion_project_dir_fpath=None)
"""<<< Creating slected data star"""

"""Finishing up >>>"""
# See the data_pipeline_nodes table in the default_pipeline.star file of any relion project directory for examples.

print('Creating RELION_OUTPUT_NODES star file ...')
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
    "{} ParticlesData.star.relion \n".format(os.path.join(outargs_rpath, output_selected_data_star_file_basename)))
relion_output_nodes_star_file.write(
    "{} DensityMap.mrc \n".format(os.path.join(outargs_rpath, output_selected_map_mrc_file_basename)))
# relion_output_nodes_star_file.write(logfile+" 13")
relion_output_nodes_star_file.write("\n")
relion_output_nodes_star_file.close()

relion_job_exit_status_file = open(os.path.join(outargs_rpath, "RELION_JOB_EXIT_SUCCESS"), "w")
relion_job_exit_status_file.close()

print('[GTF_DEBUG] Done')
"""<<< Finishing up"""
