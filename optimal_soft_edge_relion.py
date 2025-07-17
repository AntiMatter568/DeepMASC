import argparse
import os
import time
from pathlib import Path
from eval_refinement_mask import evaluate_refinement_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create soft edge masks for .mrc files.")
    parser.add_argument("-i", "--input_map_path", type=str, required=True, help="The input .mrc binary mask map file path")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="The output folder")
    parser.add_argument("-e", "--extend_inimask", type=int, default=2, help="Extend initial mask, default is 2")
    parser.add_argument("--half_map", type=str, required=True, help="Path to the half map for postprocessing")
    parser.add_argument(
        "-j", "--n_threads", type=int, default=os.cpu_count(), help="The number of threads to use, default is the number of all CPU cores"
    )
    args = parser.parse_args()

    # Create main output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Define soft edge widths to loop through
    soft_edge_widths = [5, 10, 15, 20, 25, 30]

    # Check if input files exist
    if not os.path.isfile(args.input_map_path):
        print(f"Input mask file not found: {args.input_map_path}")
        exit(1)

    if not os.path.isfile(args.half_map):
        print(f"Half map file not found: {args.half_map}")
        exit(1)

    print(f"Processing input mask: {args.input_map_path}")
    print(f"Using half map: {args.half_map}")

    # Loop through each soft edge width
    for soft_edge_width in soft_edge_widths:
        print(f"\nProcessing with soft edge width: {soft_edge_width}")

        # Create subfolder for this soft edge width
        width_output_folder = os.path.join(args.output_folder, f"soft_edge_{soft_edge_width}")
        if not os.path.exists(width_output_folder):
            os.makedirs(width_output_folder)

        # Get the base filename without extension
        emdid = Path(args.input_map_path).stem
        output_mrc = os.path.join(width_output_folder, f"{emdid}_soft_edge_{soft_edge_width}.mrc")

        if os.path.exists(output_mrc):
            print(f"Skipping {emdid} (width {soft_edge_width}) because it already exists")
            continue

        # Create the relion command
        cmd = (
            f"`which relion_mask_create` --i {args.input_map_path} --o {output_mrc} "
            f"--ini_threshold 0.01 --extend_inimask {args.extend_inimask} "
            f"--width_soft_edge {soft_edge_width} --j {args.n_threads}"
        )

        # Execute the mask creation command
        print(f"Creating mask: {cmd}")
        os.system(cmd)
        time.sleep(0.05)

        # Run postprocessing with the created mask
        postprocess_output_dir = os.path.join(width_output_folder, f"{emdid}_postprocessed")
        postprocess_cmd = (
            f"`which relion_postprocess` --mask {output_mrc} --i {args.half_map} "
            f"--o {postprocess_output_dir} --angpix -1 --auto_bfac --autob_lowres 10"
        )
        print(f"Postprocessing: {postprocess_cmd}")
        os.system(postprocess_cmd)
        time.sleep(0.05)

        eval_output_dir = os.path.join(width_output_folder, "eval_output")
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Evaluate the refinement mask using the postprocessed star file
        star_file = os.path.join(postprocess_output_dir, f"{emdid}_postprocessed.star")
        if os.path.exists(star_file):
            print(f"Evaluating refinement mask for soft edge width {soft_edge_width}")
            try:
                results = evaluate_refinement_mask(star_file, eval_output_dir)
                if results:
                    status = "PASS" if results["criterion_met"] else "FAIL"
                    print(f"Mask evaluation result: {status}")
                    print(f"  - Unmasked res (FSC=0.5): {results['unmasked_res_0_5']:.3f} Å")
                    print(f"  - Phase rand zero crossing: {results['phase_rand_zero_res']:.3f} Å")
                    print(f"  - Corrected res (FSC=0.143): {results['corrected_res_0_143']:.3f} Å")
                else:
                    print(f"Failed to evaluate mask for soft edge width {soft_edge_width}")
            except Exception as e:
                print(f"Error evaluating mask for soft edge width {soft_edge_width}: {e}")
        else:
            print(f"Warning: Star file not found: {star_file}")

    print("\nProcessing complete!")
