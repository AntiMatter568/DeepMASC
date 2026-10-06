# Optimal soft edge search

This tool finds the narrowest soft edge width, in pixels, at which a binary mask passes the PRFSC criteria described in `README_MaskEvaluation.md`. It takes a binary mask and the two half maps of a refinement and can be run from the command line or as a RELION External job.

## Algorithm

Widths are tried from 5 px in 5 px steps (5, 10, 15, ...). For each width the tool

1. builds the soft mask from the binary mask,
2. runs `relion_postprocess --auto_bfac --autob_lowres 10` with that mask and the half maps,
3. evaluates the postprocess star file with `eval_refinement_mask.py`.

The search stops at the first width whose `prfsc_pass` is true. That width is the optimal soft edge.

A mask that never passes stops at the first width whose soft mask is non-zero on any of the six faces of the box. That width is still evaluated, its values are kept, and the result is flagged as having no passing width (`prfsc_pass` false, `no_passing_width` true, stop reason `box_edge`). A width larger than the box is not tried; the search then stops at the last width tried with stop reason `box_size`. This only happens for a mask that never reaches a box face, for example an empty one.

Widths that already have a `cell.json` in the output folder are not recomputed, so an interrupted search resumes where it stopped.

## Soft mask

Voxels at or above 0.01 are set to 1. A voxel outside the mask takes the value 0.5 + 0.5 cos(pi d / w) when its distance d to the mask is below the width w, and 0 otherwise. This is the mask that `relion_mask_create --ini_threshold 0.01 --extend_inimask e --width_soft_edge w` writes, including RELION's placeholder distance of sqrt(9999) px, which changes the result for widths of 100 px and more. The default engine computes it with a distance transform, so the cost does not depend on the width. The `relion` engine runs `relion_mask_create` instead. The soft mask keeps the voxel size, origin, start indices and axis order of the input mask.

The mask extension is a single option, default 0 px. A positive value sets to 1 every voxel strictly closer than that many pixels to the mask before the soft edge is measured.

## Command line

```bash
python optimal_soft_edge_relion.py \
    -i mask.mrc -o output_dir \
    --half_map run_half1_class001_unfil.mrc
```

| Option | Description |
|--------|-------------|
| `-i`, `--input_map_path` | Binary mask MRC |
| `-o`, `--output_folder` | Output folder |
| `--half_map` | Half map 1; half map 2 is found by replacing `_half1` with `_half2` (or `_halfA` with `_halfB`) |
| `--half_map_2` | Half map 2, when its name does not follow that pattern |
| `-e`, `--extend_inimask` | Mask extension in px, default 0 |
| `--engine` | `distance` (default) or `relion` |
| `--relion_bin` | Directory with the RELION binaries, default: found on PATH |
| `--angpix` | Pixel size for `relion_postprocess`, default: read from the half map header |
| `-j`, `--n_threads` | Threads for `relion_mask_create` (engine `relion`) |
| `--reference_fsc` | `masked` (default) or `unmasked`: the FSC on which the reference crossing of clause 1 is read |
| `--reference_threshold` | `0.143` (default) or `0.5`: the threshold of the reference crossing |
| `--margin_shells` | Whole shells, default 0: the phase randomized zero must lie at least this many shells coarser than the reference crossing |

The last three options set clause 1 of the PRFSC criterion; their meaning is in `README_MaskEvaluation.md`. **The defaults (masked FSC, 0.143, 0 shells) are the paper's criteria.** The search returns the narrowest width that passes under the chosen criterion, so a stricter setting can move the optimal width or leave no passing width. A `cell.json` written under a different criterion is not reused when a search is resumed.

The RELION External job `gtf_relion4_run_optimal_soft_edge.py` takes the same options (`-i` is the mask, `-o` the job directory).

## Outputs

In the output folder:

| File | Description |
|------|-------------|
| `soft_edge_<w>/` | One folder per width tried: postprocess star file, `eval_output/` and `cell.json` |
| `<name>_optimal_mask_summary.json` | The result described below |
| `<name>_all_parameter_results.csv` | One row per width tried |
| `soft_edge_<w>/<name>_soft_edge_<w>.mrc` | The chosen soft mask; the soft masks and maps of the other widths are deleted |

The RELION job also writes `optimal_mask.mrc` (a copy of the chosen soft mask) and `optimal_soft_edge_summary.star`.

The summary has `optimal_soft_edge_width`, `prfsc_pass`, `no_passing_width`, `stop_reason` (`pass`, `box_edge` or `box_size`), `optimal_extend_inimask`, `engine`, the criterion options (`reference_fsc`, `reference_threshold`, `margin_shells`), the reference crossing at the chosen width (`optimal_reference_res`), the masked 0.143, phase randomized zero, masked zero and corrected 0.143 resolutions at the chosen width, and `widths_tried`, which lists every width with its verdict, `touches_box` and resolutions. The command-line report prints the same table.
