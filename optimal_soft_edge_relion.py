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
    print("\n" + "=" * 60)
    print("POST-PROCESSING: Combining results and selecting optimal mask")
    print("=" * 60)

    all_results = []

    # Find all folders that match the pattern "extend_X_soft_edge_Y"
    param_folders = []
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if (
            os.path.isdir(item_path)
            and item.startswith("extend_")
            and "soft_edge_" in item
        ):
            param_folders.append(item)

    if not param_folders:
        print("No parameter combination folders found!")
        return None

    print(
        f"Found {len(param_folders)} parameter combination folders: {sorted(param_folders)}"
    )

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
        csv_pattern = os.path.join(
            eval_folder, f"mask3d_evaluation_{emdid}_postprocessed.csv"
        )

        if os.path.exists(csv_pattern):
            try:
                df = pd.read_csv(csv_pattern)
                df["extend_inimask"] = extend_inimask
                df["soft_edge_width"] = soft_edge_width
                df["source_file"] = csv_pattern
                all_results.append(df)
                print(
                    f"Found results for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}"
                )
            except Exception as e:
                print(
                    f"Error reading CSV for extend_inimask {extend_inimask}, width {soft_edge_width}: {e}"
                )
        else:
            print(
                f"No CSV found for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}"
            )

    if not all_results:
        print("No valid results found! Please check the input files and parameters.")
        return None

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save combined results
    combined_csv_path = os.path.join(
        output_folder, f"{emdid}_all_parameter_results.csv"
    )
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined results saved to: {combined_csv_path}")

    valid_results = pd.DataFrame(
        combined_df[
            (combined_df["valid"] == True) & (combined_df["criterion_met"] == True)
        ]
    )

    if "phase_rand_noise_floor" in combined_df.columns:
        valid_results = pd.DataFrame(
            valid_results[valid_results["phase_rand_noise_floor"] < 0.1]
        )

    print(f"\nTotal results: {len(combined_df)}")
    print(f"Valid results meeting all criteria: {len(valid_results)}")

    if len(valid_results) == 0:
        print("No results meet all criteria!")
        print("\nFalling back to valid + criterion_met only:")
        fallback_results = pd.DataFrame(
            combined_df[
                (combined_df["valid"] == True) & (combined_df["criterion_met"] == True)
            ]
        )
        if len(fallback_results) == 0:
            fallback_results = pd.DataFrame(combined_df[combined_df["valid"] == True])
        if len(fallback_results) > 0:
            optimal_result = fallback_results.loc[
                fallback_results["corrected_res_0_143"].idxmin()  # type: ignore[union-attr]
            ]
            print(
                f"Best fallback result: extend_inimask {optimal_result['extend_inimask']}, soft edge width {optimal_result['soft_edge_width']}"
            )
        else:
            print("No valid results found at all!")
            return None
    else:
        # Primary: minimize corrected_res_0_143 (best resolution)
        # Tiebreaker: minimize correction_magnitude (most reliable mask)
        if "correction_magnitude" in valid_results.columns:
            best_resolution = valid_results["corrected_res_0_143"].min()
            resolution_tolerance = 0.05
            near_optimal = pd.DataFrame(
                valid_results[
                    valid_results["corrected_res_0_143"]
                    <= best_resolution + resolution_tolerance
                ]
            )
            optimal_result = near_optimal.loc[
                near_optimal["correction_magnitude"].idxmin()  # type: ignore[union-attr]
            ]
        else:
            optimal_result = valid_results.loc[
                valid_results["corrected_res_0_143"].idxmin()  # type: ignore[union-attr]
            ]
        print(
            f"Optimal parameters: extend_inimask {optimal_result['extend_inimask']}, soft edge width {optimal_result['soft_edge_width']}"
        )

    print(f"\nOPTIMAL MASK SUMMARY:")
    print(f"   Extend Inimask: {optimal_result['extend_inimask']}")
    print(f"   Soft Edge Width: {optimal_result['soft_edge_width']}")
    print(f"   Unmasked res (FSC=0.5): {optimal_result['unmasked_res_0_5']:.3f} Å")
    print(
        f"   Unmasked res (FSC=0.143): {optimal_result.get('unmasked_res_0_143', float('nan')):.3f} Å"
    )
    print(
        f"   Masked res (FSC=0.143): {optimal_result.get('masked_res_0_143', float('nan')):.3f} Å"
    )
    print(
        f"   Corrected res (FSC=0.143): {optimal_result['corrected_res_0_143']:.3f} Å"
    )
    print(
        f"   Correction magnitude: {optimal_result.get('correction_magnitude', float('nan')):.3f} Å"
    )
    print(
        f"   Phase rand noise floor: {optimal_result.get('phase_rand_noise_floor', float('nan')):.4f}"
    )
    print(f"   Phase rand zero crossing: {optimal_result['phase_rand_zero_res']:.3f} Å")
    print(f"   Criterion met: {'YES' if optimal_result['criterion_met'] else 'NO'}")
    print(f"   Valid: {'YES' if optimal_result['valid'] else 'NO'}")

    print(f"\nCOMPARISON TABLE:")
    comparison_cols = [
        "extend_inimask",
        "soft_edge_width",
        "corrected_res_0_143",
    ]
    optional_cols = [
        "unmasked_res_0_143",
        "masked_res_0_143",
        "correction_magnitude",
        "phase_rand_noise_floor",
    ]
    for col in optional_cols:
        if col in combined_df.columns:
            comparison_cols.append(col)
    comparison_cols.extend(["criterion_met", "valid"])

    comparison_df = combined_df[comparison_cols].sort_values(  # type: ignore[call-overload]
        ["extend_inimask", "soft_edge_width"]
    )
    for col in [
        "corrected_res_0_143",
        "unmasked_res_0_143",
        "masked_res_0_143",
        "correction_magnitude",
    ]:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(3)
    if "phase_rand_noise_floor" in comparison_df.columns:
        comparison_df["phase_rand_noise_floor"] = comparison_df[
            "phase_rand_noise_floor"
        ].round(4)
    print(comparison_df.to_string(index=False))

    summary = {
        "optimal_extend_inimask": int(optimal_result["extend_inimask"]),
        "optimal_soft_edge_width": int(optimal_result["soft_edge_width"]),
        "optimal_corrected_resolution": float(optimal_result["corrected_res_0_143"]),
        "optimal_unmasked_res_0_143": float(
            optimal_result.get("unmasked_res_0_143", float("nan"))
        ),
        "optimal_masked_res_0_143": float(
            optimal_result.get("masked_res_0_143", float("nan"))
        ),
        "optimal_correction_magnitude": float(
            optimal_result.get("correction_magnitude", float("nan"))
        ),
        "optimal_phase_rand_noise_floor": float(
            optimal_result.get("phase_rand_noise_floor", float("nan"))
        ),
        "total_results": len(combined_df),
        "valid_results": len(valid_results),
        "criterion_met": bool(optimal_result["criterion_met"]),
        "emdid": emdid,
    }

    summary_path = os.path.join(output_folder, f"{emdid}_optimal_mask_summary.json")
    import json

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return summary


def run_soft_edge_grid_search(
    input_map_path: str,
    output_folder: str,
    half_map: str,
    extend_inimask_values: list[int],
    soft_edge_widths: list[int],
    n_threads: int | None = None,
) -> dict | None:
    """
    Run grid search over extend_inimask × soft_edge_width parameter space.

    For each combination:
      1. Create soft-edged mask via relion_mask_create
      2. Run relion_postprocess with the mask and half maps
      3. Evaluate the mask quality using FSC criteria

    After all combinations are evaluated, selects the optimal parameters
    that minimize corrected_res_0_143 while meeting validity criteria.

    Args:
        input_map_path: Path to the input binary mask MRC file
        output_folder: Base output folder for all parameter combination results
        half_map: Path to half map 1 (half map 2 auto-detected via _half1→_half2 / _halfA→_halfB)
        extend_inimask_values: List of extend_inimask values to search (in pixels)
        soft_edge_widths: List of soft edge width values to search (in pixels)
        n_threads: Number of threads for relion_mask_create

    Returns:
        dict with optimal parameters and statistics, or None if no valid results found
    """
    if n_threads is None:
        n_threads = os.cpu_count() or 1

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.isfile(input_map_path):
        raise FileNotFoundError(f"Input mask file not found: {input_map_path}")
    if not os.path.isfile(half_map):
        raise FileNotFoundError(f"Half map file not found: {half_map}")

    half_map_2 = half_map.replace("_half1", "_half2").replace("_halfA", "_halfB")
    if not os.path.isfile(half_map_2):
        raise FileNotFoundError(f"Half map 2 file not found: {half_map_2}")

    emdid = Path(input_map_path).stem

    print(f"Grid search parameters:")
    print(f"  Extend inimask values: {extend_inimask_values}")
    print(f"  Soft edge widths: {soft_edge_widths}")
    print(f"  Total combinations: {len(extend_inimask_values) * len(soft_edge_widths)}")
    print(f"Processing input mask: {input_map_path}")
    print(f"Using half map: {half_map}")

    # Loop through each combination of extend_inimask and soft edge width
    for extend_inimask in extend_inimask_values:
        for soft_edge_width in soft_edge_widths:
            print(
                f"\nProcessing with extend_inimask: {extend_inimask}, soft edge width: {soft_edge_width}"
            )

            # Create subfolder for this parameter combination
            param_output_folder = os.path.join(
                output_folder, f"extend_{extend_inimask}_soft_edge_{soft_edge_width}"
            )
            os.makedirs(param_output_folder, exist_ok=True)

            output_mrc = os.path.join(
                param_output_folder,
                f"{emdid}_extend_{extend_inimask}_soft_edge_{soft_edge_width}.mrc",
            )

            if os.path.exists(output_mrc):
                print(
                    f"Skipping {emdid} (extend {extend_inimask}, width {soft_edge_width}) because it already exists"
                )
            else:
                # Create the relion command
                cmd = (
                    f"`which relion_mask_create` --i {input_map_path} --o {output_mrc} "
                    f"--ini_threshold 0.01 --extend_inimask {extend_inimask} "
                    f"--width_soft_edge {soft_edge_width} --j {n_threads}"
                )

                # Execute the mask creation command
                print(f"Creating mask: {cmd}")
                os.system(cmd)
                time.sleep(0.05)

            # Run postprocessing with the created mask
            postprocess_output_dir = os.path.join(
                param_output_folder, f"{emdid}_postprocessed"
            )
            postprocess_cmd = (
                f"`which relion_postprocess` --mask {output_mrc} --i {half_map} "
                f"--o {postprocess_output_dir} --angpix -1 --auto_bfac --autob_lowres 10"
            )
            print(f"Postprocessing: {postprocess_cmd}")
            os.system(postprocess_cmd)
            time.sleep(0.05)

            eval_output_dir = os.path.join(param_output_folder, "eval_output")
            os.makedirs(eval_output_dir, exist_ok=True)

            # Evaluate the refinement mask using the postprocessed star file
            star_file = os.path.join(param_output_folder, f"{emdid}_postprocessed.star")
            if os.path.exists(star_file):
                print(
                    f"Evaluating refinement mask for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}"
                )
                try:
                    results = evaluate_refinement_mask(star_file, eval_output_dir)
                    if results:
                        status = "PASS" if results["criterion_met"] else "FAIL"
                        print(f"Mask evaluation result: {status}")
                        print(
                            f"  - Unmasked res (FSC=0.5): {results['unmasked_res_0_5']:.3f} Å"
                        )
                        print(
                            f"  - Phase rand zero crossing: {results['phase_rand_zero_res']:.3f} Å"
                        )
                        print(
                            f"  - Corrected res (FSC=0.143): {results['corrected_res_0_143']:.3f} Å"
                        )
                    else:
                        print(
                            f"Failed to evaluate mask for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}"
                        )
                except Exception as e:
                    print(
                        f"Error evaluating mask for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}: {e}"
                    )
            else:
                print(
                    f"Warning: Star file not found for extend_inimask {extend_inimask}, soft edge width {soft_edge_width}: {star_file}"
                )

    print("\nGrid search complete!")

    optimal_summary = combine_and_select_optimal_mask(output_folder, emdid)
    return optimal_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create soft edge masks for .mrc files."
    )
    parser.add_argument(
        "-i",
        "--input_map_path",
        type=str,
        required=True,
        help="The input .mrc binary mask map file path",
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, required=True, help="The output folder"
    )
    parser.add_argument(
        "-e",
        "--extend_inimask",
        type=str,
        default="0,2,3,4",
        help="Extend initial mask values (comma-separated for grid search, e.g., '1,2,3'), default is '0,2,3,4'",
    )
    parser.add_argument(
        "--half_map",
        type=str,
        required=True,
        help="Path to the half map for postprocessing",
    )
    parser.add_argument(
        "-j",
        "--n_threads",
        type=int,
        default=os.cpu_count(),
        help="The number of threads to use, default is the number of all CPU cores",
    )
    parser.add_argument(
        "-s",
        "--soft_edge_widths",
        type=str,
        default="5,10,15,20,25",
        help="Soft edge width values (comma-separated for grid search, e.g., '5,10,15'), default is '5,10,15,20,25'",
    )
    args = parser.parse_args()

    # Parse extend_inimask values
    try:
        extend_inimask_values = [int(x.strip()) for x in args.extend_inimask.split(",")]
    except ValueError:
        print(f"Error: Invalid extend_inimask values: {args.extend_inimask}")
        exit(1)

    # Parse soft edge width values
    try:
        soft_edge_widths = [int(x.strip()) for x in args.soft_edge_widths.split(",")]
    except ValueError:
        print(f"Error: Invalid soft_edge_widths values: {args.soft_edge_widths}")
        exit(1)

    try:
        optimal_summary = run_soft_edge_grid_search(
            input_map_path=args.input_map_path,
            output_folder=args.output_folder,
            half_map=args.half_map,
            extend_inimask_values=extend_inimask_values,
            soft_edge_widths=soft_edge_widths,
            n_threads=args.n_threads,
        )
        if optimal_summary:
            print(
                f"\nFINAL RESULT: Optimal parameters are extend_inimask {optimal_summary['optimal_extend_inimask']}, soft edge width {optimal_summary['optimal_soft_edge_width']}"
            )
            print(
                f"   Best corrected resolution: {optimal_summary['optimal_corrected_resolution']:.3f} Å"
            )
        else:
            print("\nFailed to find optimal results")
            exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
