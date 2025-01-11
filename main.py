import shutil
from pathlib import Path
import subprocess
from loguru import logger
import argparse
import os
import tempfile
from map_utils import calc_map_ccc, calculate_fsc
import select

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs="+", type=str, help="List of input mrc files", required=True)
    parser.add_argument("-g", "--gpus", type=str, help="GPU ID to use for prediction", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output folder name", required=True)
    parser.add_argument("-b", "--batch", type=int, help="Batch size to use", required=False, default=4)
    parser.add_argument("--temp", type=str, help="Temporary directory path", default="/tmp")
    parser.add_argument("--debug", type=bool, help="Enable debug mode to generate full output", default=False)
    parser.add_argument("-r","--reso", choices=["Low", "High"], type=str, help="Resolution to choose the deep learning model", default="Low")
    parser.add_argument("--dryrun", action="store_true", help="Dry run, do not run CryoREAD but just print commands")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.add("AutoClass3D.log")

    # Determine resolution of model to use
    reso_input = 8.0 if args.reso == "Low" else 2.0

    logger.info("Input job folder path: ", args.files)

    CURR_SCIPT_PATH = Path(__file__).absolute().parent
    CRYOREAD_PATH = CURR_SCIPT_PATH / "CryoREAD" / "main.py"

    mrc_files = args.files
    # check if files exists
    for mrc_file in mrc_files:
        if not os.path.exists(mrc_file):
            logger.error("Input mrc file not found: " + mrc_file)
            exit(1)

    logger.info("MRC files count: " + str(len(mrc_files)))
    logger.info("MRC files path:\n" + "\n".join(mrc_files))

    # run CryoREAD

    # make temp dir
    os.makedirs(args.temp, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir=args.temp)

    try:
        temp_dir_name = temp_dir.name
        map_list = []

        for mrc_file in mrc_files:
            map_name = Path(mrc_file).stem.split(".")[0].strip()
            curr_out_dir = os.path.join(temp_dir_name, map_name)

            seg_map_path = curr_out_dir + "/input_segment.mrc"
            prot_prob_path = curr_out_dir + "/mask_protein.mrc"

            if not os.path.exists(seg_map_path) or not os.path.exists(prot_prob_path):
                logger.info(f"Running CryoREAD prediction on {mrc_file}")
                logger.info(f"MRC file: {mrc_file}")
                cmd = [
                    "python",
                    str(CRYOREAD_PATH),
                    "--mode=0",
                    f"-F={mrc_file}",
                    "--contour=0",
                    f"--gpu={args.G}",
                    f"--batch_size={args.batch}",
                    f"--prediction_only",
                    f"--resolution={reso_input}",
                    f"--output={curr_out_dir}",
                ]

                logger.info(f"Running cryoREAD command: {' '.join(cmd)}")

                if args.dryrun:
                    continue

                # Use asyncio to handle subprocess output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True,
                    env=dict(os.environ, PYTHONUNBUFFERED="1")  # Force Python subprocess to be unbuffered
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
            try:
                real_space_cc = calc_map_ccc(seg_map_path, prot_prob_path)[0]
            except:
                logger.warning("Failed to calculate real space CC")
                real_space_cc = 0.0

            try:
                x, fsc, cutoff_05, cutoff_0143 = calculate_fsc(seg_map_path, prot_prob_path)
            except:
                logger.warning("Failed to calculate FSC")
                cutoff_05 = 0.0
            map_list.append([mrc_file, real_space_cc, cutoff_05])

            if args.debug:
                final_out_path = os.path.join(args.output, map_name)
                shutil.copytree(curr_out_dir, final_out_path)
            else:
                # copyfiles to final output dir
                shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_base_prob.mrc"),
                            os.path.join(args.output, f"{map_name}_chain_base_prob.mrc"))
                shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_phosphate_prob.mrc"),
                            os.path.join(args.output, f"{map_name}_chain_phosphate_prob.mrc"))
                shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_sugar_prob.mrc"),
                            os.path.join(args.output, f"{map_name}_chain_sugar_prob.mrc"))
                shutil.copy(os.path.join(curr_out_dir, "2nd_stage_detection", "chain_protein_prob.mrc"),
                            os.path.join(args.output, f"{map_name}_chain_protein_prob.mrc"))
                shutil.copy(seg_map_path, os.path.join(args.output, f"{map_name}_segment.mrc"))
                shutil.copy(prot_prob_path, os.path.join(args.output, f"{map_name}_mask_protein.mrc"))
                shutil.copy(os.path.join(curr_out_dir, "CCC_FSC05.txt"), os.path.join(args.output, f"{map_name}_CCC_FSC05.txt"))

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
    finally:
        temp_dir.cleanup()

    if not args.dryrun:
        # sort by real space CC
        map_list.sort(key=lambda x: x[1], reverse=True)
        for idx, (mrc_file, real_space_cc, golden_standard_fsc) in enumerate(map_list):
            if idx == 0:
                logger.opt(colors=True).info(
                    "Input map: "
                    + f"<blue>{mrc_file}</blue>"
                    + ", Real space CC: "
                    + f"<blue>{real_space_cc:.4f}</blue>"
                    + ", Golden standard FSC: "
                    + f"<blue>{golden_standard_fsc:.4f}</blue>"
                )
            else:
                logger.opt(colors=True).info(
                    "Input map: "
                    + mrc_file
                    + ", Real space CC: "
                    + f"{real_space_cc:.4f}"
                    + ", Golden standard FSC: "
                    + f"{golden_standard_fsc:.4f}"
                )
