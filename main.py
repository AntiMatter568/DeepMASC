from pathlib import Path
from loguru import logger
import argparse
import os
import glob
from deepmasc_core import process_map_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=str,
        help="List of input mrc files (supports wildcards like *.mrc)",
        required=True,
    )
    parser.add_argument("-g", "--gpus", type=str, help="GPU ID to use for prediction", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output folder name", required=True)
    parser.add_argument("-b", "--batch", type=int, help="Batch size to use", required=False, default=4)
    parser.add_argument(
        "--debug",
        type=bool,
        help="Enable debug mode to generate full output",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--reso",
        choices=["Low", "High"],
        type=str,
        help="Resolution to choose the deep learning model",
        default="Low",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run, do not run CryoREAD but just print commands",
    )

    args = parser.parse_args()

    logger.info("Output folder: ", args.output)
    output_path = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)

    # Determine resolution of model to use
    reso_input = 8.0 if args.reso == "Low" else 2.0

    logger.info("Input job folder path: ", args.files)

    # Expand wildcards in file patterns
    mrc_files = []
    for pattern in args.files:
        print("found pattern: ", pattern)
        expanded_files = glob.glob(pattern)
        if expanded_files:
            mrc_files.extend(expanded_files)
        else:
            # If no files match the pattern, treat it as a literal filename
            mrc_files.append(pattern)

    # Remove duplicates while preserving order
    seen = set()
    mrc_files = [f for f in mrc_files if not (f in seen or seen.add(f))]

    # check if files exists
    for mrc_file in mrc_files:
        if not os.path.exists(mrc_file):
            logger.error("Input mrc file not found: " + mrc_file)
            exit(1)

    logger.info("MRC files count: " + str(len(mrc_files)))
    logger.info("MRC files path:\n" + "\n".join(mrc_files))

    # Process maps using shared core function
    if args.dryrun:
        logger.info("Dry run mode - skipping actual processing")
        map_list = []
    else:
        # Process all maps and get results
        # Result format: [(class_id, mrc_file, real_space_cc, cutoff_05), ...]
        result_list = process_map_files(
            mrc_files=mrc_files,
            output_path=output_path,
            gpu_ids=args.gpus,
            batch_size=args.batch,
            reso_input=reso_input,
            debug_mode=args.debug,
            class_ids=None,  # Use indices as class_ids for CLI mode
        )

        # Convert to format expected by CLI output (remove class_id)
        # CLI expects: [mrc_file, real_space_cc, golden_standard_fsc]
        map_list = [[item[1], item[2], item[3]] for item in result_list]

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
