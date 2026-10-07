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

Voxels at or above 0.01 are set to 1. A voxel outside the mask takes the value 0.5 + 0.5 cos(pi d / w) when its distance d to the mask is below the width w, and 0 otherwise. This is the mask that `relion_mask_create --ini_threshold 0.01 --extend_inimask e --width_soft_edge w` writes, including RELION's placeholder distance of sqrt(9999) px, which changes the result for widths of 100 px and more. The default engine computes it with a distance transform, so the cost does not depend on the width. The transform is computed once per mask (once more for the padded mask when the extension is above 0) and every width reuses it; the soft masks are the same as when each width computes its own. The `relion` engine runs `relion_mask_create` instead. The soft mask keeps the voxel size, origin, start indices and axis order of the input mask. The same code is available as a standalone command, see `README_SoftEdgeMask.md`.

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
| `--edt_backend` | Distance transform of engine `distance`: `scipy` (default), `taichi` or `auto` (Taichi if it starts, else scipy) |
| `--taichi_arch` | `cpu` (uses `--n_threads` threads), `cuda` or `auto` (default) |
| `--relion_bin` | Directory with the RELION binaries, default: found on PATH |
| `--angpix` | Pixel size for `relion_postprocess`, default: read from the half map header |
| `-j`, `--n_threads` | Threads for `relion_mask_create` (engine `relion`) and for the Taichi CPU backend |
| `--reference_fsc` | `masked` (default) or `unmasked`: the FSC on which the reference crossing of clause 1 is read |
| `--reference_threshold` | `0.143` (default) or `0.5`: the threshold of the reference crossing |
| `--margin_shells` | Whole shells, default 0: the phase randomized zero must lie at least this many shells coarser than the reference crossing |

The last three options set clause 1 of the PRFSC criterion; their meaning is in `README_MaskEvaluation.md`. **The defaults (masked FSC, 0.143, 0 shells) are the paper's criteria.** The search returns the narrowest width that passes under the chosen criterion, so a stricter setting can move the optimal width or leave no passing width. A `cell.json` written under a different criterion is not reused when a search is resumed.

The RELION External job `gtf_relion4_run_optimal_soft_edge.py` takes the same options (`-i` is the mask, `-o` the job directory).

## Speed

For a box of 512 voxels or more the distance transform dominates the cost of a search. The search computes it once per mask and reuses it for every width, and the cosine of the edge is evaluated only on the voxels inside the edge. The optional Taichi backend (`--edt_backend taichi`, see `README_SoftEdgeMask.md` for installation and fallback) makes the transform itself faster on several CPU threads or on a GPU. Both are optional in the sense that the result does not change: the soft masks are bit-identical with and without reuse, and the Taichi squared distances equal scipy's.

Time to build the soft masks of a 5-width search (widths 5 to 25 px), without `relion_postprocess`, in seconds. "Per width" is the code of DeepMASC main before the change (one transform and a full-array cosine for each width); "shared" is the code with the reuse. Each value is the better of two runs on a real mask. Padding 6 px adds a second transform per width (per mask when shared).

| Mask | Box | Padding | scipy, 1 thread, per width | scipy, 1 thread, shared | Taichi CPU, 16 threads, shared | Taichi CUDA, shared |
|---|---|---|---|---|---|---|
| EMD-51414 | 600 | 0 | 428 | 82 | 15.6 | 5.5 |
| EMD-51414 | 600 | 6 | 811 | 159 | 24.0 | 8.1 |
| EMD-15364 | 640 | 0 | 455 | 84 | 16.8 | 5.5 |
| EMD-15364 | 640 | 6 | 843 | 162 | 26.9 | 9.6 |

One transform of the 600 box takes 74 s with scipy, 7.9 s with Taichi on 16 CPU threads and 2.3 s on the GPU; the remainder of the shared time is the five cosines and mask arrays (about 8 s in this run). The CPU runs were on a 24-core machine with slow cores (a Pascal-era GPU host), so the scipy times are about 3 times those of a current node (a 512 box takes about 20 s per transform on a 48-core Ada host). The GPU was an RTX PRO 6000 Blackwell on another host. Single-threaded Taichi was no faster than scipy in the earlier benchmark. Timing code and raw values are not part of this repository.

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
