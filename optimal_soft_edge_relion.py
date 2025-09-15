import argparse
import os
import time
from pathlib import Path
import pandas as pd
import glob
from eval_refinement_mask import evaluate_refinement_mask


def combine_and_select_optimal_mask(output_folder, emdid):
    """
    Combine all CSV results and select the optimal parameter combination based on
    highest corrected resolution that meets criteria and is valid.
    
    Args:
        output_folder (str): Base output folder containing all parameter combination results
        emdid (str): The EMD ID from the input filename
        
    Returns:
        dict: Results summary with optimal parameters and statistics
    """
    print("\n" + "="*60)
    print("POST-PROCESSING: Combining results and selecting optimal mask")
    print("="*60)
    
    all_results = []
    
    # Find all folders that match the pattern "extend_X_soft_edge_Y"
    param_folders = []
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if os.path.isdir(item_path) and item.startswith("extend_") and "soft_edge_" in item:
            param_folders.append(item)
    
    if not param_folders:
        print("No parameter combination folders found!")
        return None
    
    print(f"Found {len(param_folders)} parameter combination folders: {sorted(param_folders)}")
    
    # Collect all CSV results
    for folder_name in param_folders:
        # Extract parameters from folder name (e.g., "extend_2_soft_edge_10" -> extend=2, width=10)
        try:
            parts = folder_name.split("_")
            extend_idx = parts.index("extend")
            soft_edge_idx = parts.index("soft")
            
            extend_inimask = int(parts[extend_idx + 1])
            soft_edge_width = int(parts[soft_edge_idx + 2])  # "soft", "edge", "width"
        except (ValueError, IndexError):
            print(f"Could not extract parameters from folder name: {folder_name}")
            continue
            
        param_folder = os.path.join(output_folder, folder_name)
        eval_folder = os.path.join(param_folder, "eval_output")
        csv_pattern = os.path.join(eval_folder, f"mask3d_evaluation_{emdid}_postprocessed.csv")
        
        if os.path.exists(csv_pattern):
            try:
                df = pd.read_csv(csv_pattern)
                df['extend_inimask'] = extend_inimask
                df['soft_edge_width'] = soft_edge_width
                df['source_file'] = csv_pattern
                all_results.append(df)
                print(f"Found results for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}")
            except Exception as e:
                print(f"Error reading CSV for extend_inimask {extend_inimask}, width {soft_edge_width}: {e}")
        else:
            print(f"No CSV found for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}")
    
    if not all_results:
        print("No valid results found! Please check the input files and parameters.")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_csv_path = os.path.join(output_folder, f"{emdid}_all_parameter_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined results saved to: {combined_csv_path}")
    
    # Filter for valid results that meet criteria
    valid_results = combined_df[
        (combined_df['valid'] == True) & 
        (combined_df['criterion_met'] == True)
    ]
    
    print(f"\nTotal results: {len(combined_df)}")
    print(f"Valid results meeting criteria: {len(valid_results)}")
    
    if len(valid_results) == 0:
        print("No results meet both validity and criterion requirements!")
        print("\nFalling back to valid results only:")
        fallback_results = combined_df[combined_df['valid'] == True]
        if len(fallback_results) > 0:
            optimal_result = fallback_results.loc[fallback_results['corrected_res_0_143'].idxmax()]
            print(f"Best valid result (no criterion check): extend_inimask {optimal_result['extend_inimask']}, soft edge width {optimal_result['soft_edge_width']}")
        else:
            print("No valid results found at all!")
            return None
    else:
        # Select the one with highest corrected resolution (lowest value = higher resolution)
        optimal_result = valid_results.loc[valid_results['corrected_res_0_143'].idxmin()]
        print(f"Optimal parameters: extend_inimask {optimal_result['extend_inimask']}, soft edge width {optimal_result['soft_edge_width']}")
    
    # Print detailed results
    print(f"\nOPTIMAL MASK SUMMARY:")
    print(f"   Extend Inimask: {optimal_result['extend_inimask']}")
    print(f"   Soft Edge Width: {optimal_result['soft_edge_width']}")
    print(f"   Unmasked res (FSC=0.5): {optimal_result['unmasked_res_0_5']:.3f} Å")
    print(f"   Phase rand zero crossing: {optimal_result['phase_rand_zero_res']:.3f} Å")
    print(f"   Corrected res (FSC=0.143): {optimal_result['corrected_res_0_143']:.3f} Å")
    print(f"   Criterion met: {'YES' if optimal_result['criterion_met'] else 'NO'}")
    print(f"   Valid: {'YES' if optimal_result['valid'] else 'NO'}")
    
    # Show comparison table
    print(f"\nCOMPARISON TABLE:")
    comparison_cols = ['extend_inimask', 'soft_edge_width', 'corrected_res_0_143', 'criterion_met', 'valid']
    comparison_df = combined_df[comparison_cols].sort_values(['extend_inimask', 'soft_edge_width'])
    comparison_df['corrected_res_0_143'] = comparison_df['corrected_res_0_143'].round(3)
    print(comparison_df.to_string(index=False))
    
    # Save summary
    summary = {
        'optimal_extend_inimask': int(optimal_result['extend_inimask']),
        'optimal_soft_edge_width': int(optimal_result['soft_edge_width']),
        'optimal_corrected_resolution': float(optimal_result['corrected_res_0_143']),
        'total_results': len(combined_df),
        'valid_results': len(valid_results),
        'criterion_met': bool(optimal_result['criterion_met']),
        'emdid': emdid
    }
    
    summary_path = os.path.join(output_folder, f"{emdid}_optimal_mask_summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create soft edge masks for .mrc files.")
    parser.add_argument("-i", "--input_map_path", type=str, required=True, help="The input .mrc binary mask map file path")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="The output folder")
    parser.add_argument("-e", "--extend_inimask", type=str, default="0,2,3,4", help="Extend initial mask values (comma-separated for grid search, e.g., '1,2,3'), default is '2'")
    parser.add_argument("--half_map", type=str, required=True, help="Path to the half map for postprocessing")
    parser.add_argument(
        "-j", "--n_threads", type=int, default=os.cpu_count(), help="The number of threads to use, default is the number of all CPU cores"
    )
    parser.add_argument(
        "-s", "--soft_edge_widths", type=str, default="5,10,15,20,25", 
        help="Soft edge width values (comma-separated for grid search, e.g., '5,10,15'), default is '5,10,15,20,25'"
    )
    args = parser.parse_args()

    # Create main output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Parse extend_inimask values
    try:
        extend_inimask_values = [int(x.strip()) for x in args.extend_inimask.split(',')]
    except ValueError:
        print(f"Error: Invalid extend_inimask values: {args.extend_inimask}")
        exit(1)

    # Parse soft edge width values
    try:
        soft_edge_widths = [int(x.strip()) for x in args.soft_edge_widths.split(',')]
    except ValueError:
        print(f"Error: Invalid soft_edge_widths values: {args.soft_edge_widths}")
        exit(1)

    print(f"Grid search parameters:")
    print(f"  Extend inimask values: {extend_inimask_values}")
    print(f"  Soft edge widths: {soft_edge_widths}")
    print(f"  Total combinations: {len(extend_inimask_values) * len(soft_edge_widths)}")

    # Check if input files exist
    if not os.path.isfile(args.input_map_path):
        print(f"Input mask file not found: {args.input_map_path}")
        exit(1)

    # Check for half map 1
    if not os.path.isfile(args.half_map):
        print(f"Half map file not found: {args.half_map}")
        exit(1)
    
    # Check for half map 2
    half_map_2 = args.half_map.replace('_half1', '_half2').replace('_halfA', '_halfB')
    if not os.path.isfile(half_map_2):
        print(f"Half map 2 file not found: {half_map_2}")
        exit(1)

    print(f"Processing input mask: {args.input_map_path}")
    print(f"Using half map: {args.half_map}")

    # Get the base filename without extension
    emdid = Path(args.input_map_path).stem

    # Loop through each combination of extend_inimask and soft edge width
    for extend_inimask in extend_inimask_values:
        for soft_edge_width in soft_edge_widths:
            print(f"\nProcessing with extend_inimask: {extend_inimask}, soft edge width: {soft_edge_width}")

            # Create subfolder for this parameter combination
            param_output_folder = os.path.join(args.output_folder, f"extend_{extend_inimask}_soft_edge_{soft_edge_width}")
            if not os.path.exists(param_output_folder):
                os.makedirs(param_output_folder)

            output_mrc = os.path.join(param_output_folder, f"{emdid}_extend_{extend_inimask}_soft_edge_{soft_edge_width}.mrc")

            if os.path.exists(output_mrc):
                print(f"Skipping {emdid} (extend {extend_inimask}, width {soft_edge_width}) because it already exists")
                continue

            # Create the relion command
            cmd = (
                f"`which relion_mask_create` --i {args.input_map_path} --o {output_mrc} "
                f"--ini_threshold 0.01 --extend_inimask {extend_inimask} "
                f"--width_soft_edge {soft_edge_width} --j {args.n_threads}"
            )

            # Execute the mask creation command
            print(f"Creating mask: {cmd}")
            os.system(cmd)
            time.sleep(0.05)

            # Run postprocessing with the created mask
            postprocess_output_dir = os.path.join(param_output_folder, f"{emdid}_postprocessed")
            postprocess_cmd = (
                f"`which relion_postprocess` --mask {output_mrc} --i {args.half_map} "
                f"--o {postprocess_output_dir} --angpix -1 --auto_bfac --autob_lowres 10"
            )
            print(f"Postprocessing: {postprocess_cmd}")
            os.system(postprocess_cmd)
            time.sleep(0.05)

            eval_output_dir = os.path.join(param_output_folder, "eval_output")
            if not os.path.exists(eval_output_dir):
                os.makedirs(eval_output_dir)

            # Evaluate the refinement mask using the postprocessed star file
            star_file = os.path.join(param_output_folder, f"{emdid}_postprocessed.star")
            if os.path.exists(star_file):
                print(f"Evaluating refinement mask for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}")
                try:
                    results = evaluate_refinement_mask(star_file, eval_output_dir)
                    if results:
                        status = "PASS" if results["criterion_met"] else "FAIL"
                        print(f"Mask evaluation result: {status}")
                        print(f"  - Unmasked res (FSC=0.5): {results['unmasked_res_0_5']:.3f} Å")
                        print(f"  - Phase rand zero crossing: {results['phase_rand_zero_res']:.3f} Å")
                        print(f"  - Corrected res (FSC=0.143): {results['corrected_res_0_143']:.3f} Å")
                    else:
                        print(f"Failed to evaluate mask for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}")
                except Exception as e:
                    print(f"Error evaluating mask for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}: {e}")
            else:
                print(f"Warning: Star file not found for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}: {star_file}")

    print("\nProcessing complete!")
    
    # Run post-processing to combine results and select optimal mask
    try:
        optimal_summary = combine_and_select_optimal_mask(args.output_folder, emdid)
        if optimal_summary:
            print(f"\nFINAL RESULT: Optimal parameters are extend_inimask {optimal_summary['optimal_extend_inimask']}, soft edge width {optimal_summary['optimal_soft_edge_width']}")
            print(f"   Best corrected resolution: {optimal_summary['optimal_corrected_resolution']:.3f} Å")
        else:
            print("\nPost-processing failed to find optimal results")
    except Exception as e:
        print(f"\nError during post-processing: {e}")
        import traceback
        traceback.print_exc()
