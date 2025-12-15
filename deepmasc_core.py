"""
Core processing logic for DeepMASC map analysis.
Shared by both CLI (main.py) and Relion (gtf_relion4_run_select_class3d.py) modes.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import select
from pathlib import Path
from loguru import logger
from map_utils import calc_map_ccc, calculate_fsc, is_map_empty


def process_map_files(
    mrc_files,
    output_path,
    gpu_ids,
    batch_size,
    reso_input,
    debug_mode=False,
    class_ids=None,
):
    """
    Core processing logic for DeepMASC map analysis.

    Args:
        mrc_files: List of MRC file paths to process
        output_path: Output directory path
        gpu_ids: GPU IDs to use for CryoREAD prediction
        batch_size: Batch size for CryoREAD prediction
        reso_input: Resolution parameter for CryoREAD (8.0 for Low, 2.0 for High)
        debug_mode: If True, copy all intermediate files to output
        class_ids: Optional list of class IDs corresponding to mrc_files.
                   If None, uses indices (0, 1, 2, ...)

    Returns:
        list: [(class_id, mrc_file, real_space_cc, cutoff_05), ...]
              Sorted by real_space_cc in descending order
    """
    # Create temp directory
    temp_path = os.path.join(output_path, "temp")
    os.makedirs(temp_path, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir=temp_path)
    temp_dir_path = os.path.abspath(temp_dir.name)

    # Get script path and CryoREAD path
    CURR_SCRIPT_PATH = Path(__file__).absolute().parent
    CRYOREAD_PATH = CURR_SCRIPT_PATH / "CryoREAD" / "main.py"

    result_list = []

    try:
        for idx, mrc_file in enumerate(mrc_files):
            # Get class_id for this map
            class_id = class_ids[idx] if class_ids is not None else idx

            # Check if map is empty
            if is_map_empty(mrc_file):
                logger.warning(f"Empty map found, skipping {mrc_file}")
                result_list.append([class_id, mrc_file, 0.0, 0.0])
                continue

            # Prepare paths
            map_name = Path(mrc_file).stem.split(".")[0].strip()
            curr_out_dir = os.path.join(temp_dir_path, map_name)
            os.makedirs(curr_out_dir, exist_ok=True)

            seg_map_path = os.path.join(curr_out_dir, "input_segment.mrc")
            prot_prob_path = os.path.join(curr_out_dir, "mask_protein.mrc")

            # Run CryoREAD if outputs don't exist
            if not os.path.exists(seg_map_path) or not os.path.exists(prot_prob_path):
                logger.info(f"Running CryoREAD prediction on {mrc_file}")

                cmd = [
                    sys.executable,
                    str(CRYOREAD_PATH),
                    "--mode=0",
                    f"-F={mrc_file}",
                    "--contour=0",
                    f"--gpu={gpu_ids}",
                    f"--batch_size={batch_size}",
                    "--prediction_only",
                    f"--resolution={reso_input}",
                    f"--output={curr_out_dir}",
                ]

                logger.info(f"Running CryoREAD command: {' '.join(cmd)}")

                # Run subprocess with real-time output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"),
                )

                # Read and print output
                outputs = [process.stdout, process.stderr]
                while outputs:
                    readable, _, _ = select.select(outputs, [], [])
                    for output in readable:
                        line = output.readline()
                        if not line:
                            outputs.remove(output)
                            continue
                        if output == process.stdout:
                            logger.info(line.strip())
                        else:
                            logger.error(line.strip())

                # Wait for process to complete
                process.wait()

                if process.returncode != 0:
                    raise RuntimeError(f"CryoREAD failed with exit code {process.returncode}")

            # Calculate metrics
            try:
                real_space_cc = calc_map_ccc(seg_map_path, prot_prob_path)[0]
            except Exception as e:
                logger.warning(f"Failed to calculate real space CC: {e}")
                real_space_cc = 0.0

            try:
                fsc_output_path = os.path.join(curr_out_dir, "fsc_data.txt")
                x, fsc, cutoff_05, cutoff_0143 = calculate_fsc(seg_map_path, prot_prob_path, fsc_output_path)
            except Exception as e:
                logger.warning(f"Failed to calculate FSC: {e}")
                cutoff_05 = 0.0

            # Add to results
            result_list.append([class_id, mrc_file, real_space_cc, cutoff_05])

            # Copy files to final output directory
            if debug_mode:
                # Copy everything
                final_out_path = os.path.join(output_path, map_name)
                shutil.copytree(curr_out_dir, final_out_path)
            else:
                # Copy specific files including FSC plots
                files_to_copy = [
                    ("2nd_stage_detection/chain_base_prob.mrc", f"{map_name}_chain_base_prob.mrc"),
                    ("2nd_stage_detection/chain_phosphate_prob.mrc", f"{map_name}_chain_phosphate_prob.mrc"),
                    ("2nd_stage_detection/chain_sugar_prob.mrc", f"{map_name}_chain_sugar_prob.mrc"),
                    ("2nd_stage_detection/chain_protein_prob.mrc", f"{map_name}_chain_protein_prob.mrc"),
                    ("input_segment.mrc", f"{map_name}_segment.mrc"),
                    ("mask_protein.mrc", f"{map_name}_mask_protein.mrc"),
                    ("CCC_FSC05.txt", f"{map_name}_CCC_FSC05.txt"),
                    ("fsc_data.txt", f"{map_name}_fsc_data.txt"),
                    ("fsc_data_plot.png", f"{map_name}_fsc_data_plot.png"),
                ]

                for src_rel, dst_name in files_to_copy:
                    src_path = os.path.join(curr_out_dir, src_rel)
                    dst_path = os.path.join(output_path, dst_name)
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dst_path)
                    else:
                        logger.warning(f"File not found, skipping: {src_path}")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

    finally:
        # Cleanup temp directory after all files are copied
        temp_dir.cleanup()
        logger.info("Temporary directory cleaned up")

    # Sort by real_space_cc (descending)
    result_list.sort(key=lambda x: x[2], reverse=True)

    return result_list
