from pathlib import Path
import subprocess
from glob import glob
from loguru import logger
import argparse
import os
from map_utils import calc_map_ccc, calculate_fsc

if __name__ == "__main__":

    logger.add("AutoClass3D.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("-F", nargs="+", type=str, help="List of input mrc files", required=True)
    parser.add_argument("-G", type=str, help="GPU ID to use for prediction", required=False, default="")
    parser.add_argument("-J", type=str, help="Job name / output folder name", required=True)

    args = parser.parse_args()

    logger.info("Input job folder path: ", args.F)

    CRYOREAD_PATH = "./CryoREAD/main.py"

    # mrc_files = glob(args.F + "/*.mrc")

    mrc_files = args.F
    # check if files exists
    for mrc_file in mrc_files:
        if not os.path.exists(mrc_file):
            logger.error("Input mrc file not found: " + mrc_file)
            exit(1)

    OUTDIR = str(Path("./CryoREAD_Predict_Result").absolute() / args.J)
    os.makedirs(OUTDIR, exist_ok=True)

    logger.info("MRC files count: " + str(len(mrc_files)))
    logger.info("MRC files path:\n" + "\n".join(mrc_files))

    # run CryoREAD

    map_list = []

    for mrc_file in mrc_files:

        curr_out_dir = OUTDIR + "/" + Path(mrc_file).stem.split(".")[0]

        seg_map_path = curr_out_dir + "/input_segment.mrc"
        prot_prob_path = curr_out_dir + "/mask_protein.mrc"

        if not os.path.exists(seg_map_path) or not os.path.exists(prot_prob_path):
            logger.info(f"Running CryoREAD prediction on {mrc_file}")
            cmd = [
                "python",
                CRYOREAD_PATH,
                "--mode=0",
                f"-F={mrc_file}",
                "--contour=0",
                f"--gpu={args.G}",
                f"--batch_size=4",
                f"--prediction_only",
                f"--resolution=8.0",
                f"--output={curr_out_dir}",
            ]
            process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       universal_newlines=True)

            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    logger.info(output.strip())  # Log stdout

            rc = process.poll()
            while True:
                err = process.stderr.readline()
                if err == "" and process.poll() is not None:
                    break
                if err:
                    logger.error(err.strip())  # Log stderr
        try:
            real_space_cc = calc_map_ccc(seg_map_path, prot_prob_path)[0]
        except:
            logger.warning("Failed to calculate real space CC, maybe the map is empty")
            real_space_cc = 0.0

        try:
            x, fsc, cutoff_05, cutoff_0143 = calculate_fsc(seg_map_path, prot_prob_path)
        except:
            logger.warning("Failed to calculate FSC, maybe the map is empty")
            cutoff_05 = 0.0
        map_list.append([mrc_file, real_space_cc, cutoff_05])

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
