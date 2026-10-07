# Soft edge mask

`soft_edge_mask.py` turns a binary mask into a soft mask: voxels at or above the threshold become 1, and a voxel outside the mask takes the value 0.5 + 0.5 cos(pi d / w) when its distance d to the mask is below the width w, and 0 otherwise. The result equals what `relion_mask_create --ini_threshold T --extend_inimask E --width_soft_edge W` writes, to float32 rounding, and does not need RELION. The same code builds the soft masks of the optimal soft edge search (`README_OptimalSoftEdge.md`).

## Usage

```bash
python soft_edge_mask.py -i mask.mrc -o soft.mrc --width_px 8
python soft_edge_mask.py -i mask.mrc -o soft.mrc --width_A 12 --extend_A 6
python soft_edge_mask.py -i mask.mrc -o soft.mrc --preset cryosparc_v5 --resolution 4.5
```

| Option | Meaning |
|---|---|
| `-i`, `-o` | Binary mask in, soft mask out (MRC, float32) |
| `--width_px W` or `--width_A W` | Soft edge width (RELION `--width_soft_edge`). Required unless a preset is given |
| `--extend_px E` or `--extend_A E` | Padding (RELION `--extend_inimask`), default 0. Voxels strictly closer than E to the mask are set to 1 before the edge is measured |
| `--ini_threshold T` | Voxels at or above T are the binary mask, default 0.01 |
| `--preset NAME` | `cryosparc_v4` or `cryosparc_v5`, see below |
| `--resolution R` | R in Å, only for `cryosparc_v5` |

The output keeps the voxel size, origin, start indices (`nxstart`, `nystart`, `nzstart`) and axis order of the input.

The tool prints the width and padding in pixels and Å, the threshold, the voxel size and whether the soft mask touches a box face. A soft mask that is non-zero on a face of the box is clipped by the box; the tool then prints a warning and still writes the mask.

## Units

Width and padding are each given in pixels or in Å, not both. Å values are divided by the voxel size of the input mask, which must be the same along the three axes. Fractional pixel values are kept, as RELION accepts them: 5.25 Å at 1.5 Å/px is a width of 3.5 px. The width must be positive and the padding not negative; RELION's negative padding (mask shrinking) is not supported. Conflicting options are refused and nothing is written.

## cryoSPARC presets

| Preset | Padding | Edge |
|---|---|---|
| `cryosparc_v4` | 6 Å | 6 Å |
| `cryosparc_v5 --resolution R` | 2R | 3R |

`cryosparc_v4` is the soft-edge rule of every cryoSPARC version before v5.0, `cryosparc_v5` that of v5.0. A preset cannot be combined with an explicit width or padding, and `cryosparc_v5` without `--resolution` is refused.

The presets apply cryoSPARC's soft-edge rule to the binary mask you give. cryoSPARC's own threshold step (a fraction of the map maximum) and its auto-tightening are not part of it, so the result is not the mask cryoSPARC itself would make from a map.

On the voxel grid, as `relion_mask_create` does it, the padding is the set of voxel centres strictly closer than the padding distance, and the edge is measured from the padded mask. Along an axis the mask is therefore 1 out to the last voxel centre inside the padding, and reaches 0 up to one voxel short of padding + edge.

## Equivalence with RELION

The soft mask is computed with a Euclidean distance transform instead of RELION's cube search, and reproduces RELION 5.0.1 `autoMask`, including its initial squared distance of 9999 (which matters for widths of 100 px and more). Against `relion_mask_create` on four generated low-resolution masks (boxes 100 to 144, 6 cases each: pixel width, Å width, padding in px and in Å, both presets), the largest absolute difference is 6e-8 and no voxel differs by more than 1e-6. The cost of the distance transform does not depend on the width. On a box of 100 the command takes about 1 s at widths of 3, 10 and 30 px; `relion_mask_create` with 8 threads takes 0.13, 1.8 and 34 s.

## Distance transform backend

The soft edge needs the Euclidean distance transform of the mask (twice with padding). The default backend is `scipy.ndimage.distance_transform_edt`. An optional backend written in [Taichi](https://www.taichi-lang.org/) (`soft_edge_taichi.py`) computes the same squared distances on CPU threads or on a CUDA GPU.

| Option | Description |
|---|---|
| `--edt_backend` | `scipy` (default), `taichi`, or `auto` |
| `--taichi_arch` | `cpu`, `cuda`, or `auto` (default: CUDA when a device is usable, else CPU) |

- `taichi` fails with an error message when the `taichi` package is missing, does not start, or (with `--taichi_arch cuda`) finds no CUDA device.
- `auto` uses Taichi when it starts and otherwise prints a note and uses scipy.
- The squared distances are the same integers as scipy's on every voxel. The soft mask is the same function of them; on the test shapes the two backends agree exactly, and the tolerance of the tests is 1e-6.
- The Taichi CPU backend uses all cores of the machine. Set the number of threads in the library with `threads` (the search passes its `--n_threads`).

Taichi is not a dependency of DeepMASC. Install it with `pip install taichi`; version 1.7.x was tested (Python 3.11). Taichi publishes wheels for a limited range of Python versions, so check the wheel list of the release for the Python in use. CUDA needs a driver recent enough for the card.

Taichi gains nothing on one CPU thread and little on small boxes (a few seconds with scipy); it is meant for boxes of 512 voxels and more with several CPU threads or a GPU. Measured times are in `README_OptimalSoftEdge.md`.

In the library, `capped_squared_distance(data, ...)` returns the binary mask and the squared distance capped at 9999 (one transform, two with padding), `soft_edge_from_distance(binary, r2, width)` turns it into the soft mask of one width, and `SoftEdgeDistance(src, ...)` writes the soft masks of several widths from one mask file with one transform. `relion_soft_edge` and `write_soft_mask` take the same backend options.

## RELION External job

```
python /path/to/gtf_relion4_run_soft_edge_mask.py --width_A 12
```

Set the input as a mask (`--in_mask`) and give the same width, padding, threshold, preset and backend options as above. The job directory receives:

| File | Content |
|---|---|
| `soft_mask.mrc` | The soft mask (output node, `Mask3D.mrc`) |
| `soft_edge_mask_summary.star` | Width and padding in px and Å, voxel size, threshold, box-face flag (output node, `LogFile.star`) |
| `RELION_OUTPUT_NODES.star` | The two output nodes |
| `RELION_JOB_EXIT_SUCCESS` | Written when the job finishes; `RELION_JOB_EXIT_FAILURE` is written instead when an option is refused or the job fails |
