import argparse
import glob
import json
import os
import subprocess

import mrcfile
import numpy as np
import pandas as pd

from eval_refinement_mask import evaluate_refinement_mask
from soft_edge_mask import INI_THRESHOLD, touches_box, write_soft_mask

START_WIDTH = 5
STEP_WIDTH = 5

RESULT_COLUMNS = [
    "prfsc_pass",
    "masked_res_0_143",
    "phase_rand_zero_res",
    "masked_zero_res",
    "corrected_res_0_143",
]


def _tool(name, relion_bin):
    return os.path.join(relion_bin, name) if relion_bin else name


def create_soft_mask(
    input_map_path, output_mrc, width, extend_inimask, engine, n_threads, relion_bin=None
):
    """Write the soft mask for one width: distance transform (default) or relion_mask_create."""
    if engine == "distance":
        write_soft_mask(input_map_path, output_mrc, width, extend=extend_inimask)
        return
    cmd = [
        _tool("relion_mask_create", relion_bin),
        "--i", input_map_path,
        "--o", output_mrc,
        "--ini_threshold", str(INI_THRESHOLD),
        "--extend_inimask", str(extend_inimask),
        "--width_soft_edge", str(width),
        "--j", str(n_threads),
    ]  # fmt: skip
    print(f"Creating mask: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_postprocess(mask, half1, out_root, angpix, relion_bin=None):
    """relion_postprocess with the soft mask; writes <out_root>.star."""
    cmd = [
        _tool("relion_postprocess", relion_bin),
        "--mask", mask,
        "--i", half1,
        "--o", out_root,
        "--angpix", str(angpix),
        "--auto_bfac",
        "--autob_lowres", "10",
    ]  # fmt: skip
    print(f"Postprocessing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return os.path.exists(out_root + ".star")


def _link(src, dst):
    """Symlink the half map under the name relion_postprocess expects (half 2 is found by name)."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def _box_size(path):
    with mrcfile.open(path, permissive=True, header_only=True) as f:
        h = f.header
        return int(max(h.nx, h.ny, h.nz))  # type: ignore[union-attr]


def _evaluate_width(
    width, input_map_path, half1, emdid, output_folder, extend_inimask, engine,
    n_threads, relion_bin, angpix,
):  # fmt: skip
    """Soft mask, postprocess and evaluation for one width; the result is kept in cell.json."""
    width_dir = os.path.join(output_folder, f"soft_edge_{width}")
    cell_json = os.path.join(width_dir, "cell.json")
    if os.path.exists(cell_json):
        print(f"Skipping soft edge width {width}: already evaluated")
        with open(cell_json) as f:
            return json.load(f)
    os.makedirs(width_dir, exist_ok=True)

    mask_mrc = os.path.join(width_dir, f"{emdid}_soft_edge_{width}.mrc")
    create_soft_mask(
        input_map_path, mask_mrc, width, extend_inimask, engine, n_threads, relion_bin
    )
    with mrcfile.open(mask_mrc, permissive=True) as f:
        touched = touches_box(np.asarray(f.data))

    out_root = os.path.join(width_dir, f"{emdid}_postprocessed")
    star_file = out_root + ".star"
    if not run_postprocess(mask_mrc, half1, out_root, angpix, relion_bin):
        raise RuntimeError(f"relion_postprocess wrote no star file for width {width}")
    eval_output_dir = os.path.join(width_dir, "eval_output")
    results = evaluate_refinement_mask(star_file, eval_output_dir)
    if results is None:
        raise RuntimeError(f"Could not evaluate {star_file}")

    cell = {"width": width, "touches_box": touched}
    cell.update({k: results[k] for k in RESULT_COLUMNS})
    cell.update({"prfsc_pass": bool(results["prfsc_pass"]), "valid": bool(results["valid"])})
    with open(cell_json, "w") as f:
        json.dump(cell, f, indent=2)
    return cell


def run_soft_edge_search(
    input_map_path: str,
    output_folder: str,
    half_map: str,
    extend_inimask: float = 0,
    engine: str = "distance",
    n_threads: int | None = None,
    relion_bin: str | None = None,
    half_map_2: str | None = None,
    angpix: float | None = None,
) -> dict:
    """
    Search for the optimal soft edge: the narrowest width (px) at which the mask passes the PRFSC
    criteria.

    Widths 5, 10, 15 ... px are tried in order. For each: soft mask, relion_postprocess with the
    half maps, evaluation (eval_refinement_mask). The search stops at the first width that passes.
    A mask that never passes stops at the first width whose soft mask is non-zero on a box face
    (that width is evaluated, kept and flagged "no passing width", stop reason "box_edge"). A width
    larger than the box is not tried (stop reason "box_size"). Widths with a cell.json from an
    earlier run are not recomputed.

    Args:
        input_map_path: Path to the input binary mask MRC file
        output_folder: Output folder; one soft_edge_<width> subfolder per width tried
        half_map: Path to half map 1
        extend_inimask: Mask extension in pixels applied before the soft edge (default 0)
        engine: "distance" (distance transform, same values as relion_mask_create) or "relion"
        n_threads: Number of threads for relion_mask_create
        relion_bin: Directory holding the RELION binaries (default: found on PATH)
        half_map_2: Path to half map 2 (default: half 1 name with _half1 -> _half2 / _halfA -> _halfB)
        angpix: Pixel size for relion_postprocess (default: read from the half map header)

    Returns:
        dict with the chosen width, the verdict, the stop reason and every width tried
    """
    if engine not in ("distance", "relion"):
        raise ValueError(f"engine must be 'distance' or 'relion', not {engine!r}")
    if n_threads is None:
        n_threads = os.cpu_count() or 1
    if not os.path.isfile(input_map_path):
        raise FileNotFoundError(f"Input mask file not found: {input_map_path}")
    if not os.path.isfile(half_map):
        raise FileNotFoundError(f"Half map file not found: {half_map}")
    if half_map_2 is None:
        half_map_2 = half_map.replace("_half1", "_half2").replace("_halfA", "_halfB")
    if not os.path.isfile(half_map_2):
        raise FileNotFoundError(f"Half map 2 file not found: {half_map_2}")

    os.makedirs(output_folder, exist_ok=True)
    emdid = os.path.splitext(os.path.basename(input_map_path))[0]
    # relion_postprocess takes half 1 and finds half 2 by name, so both are linked under such names
    half1 = os.path.join(output_folder, f"{emdid}_half1.map")
    _link(half_map, half1)
    _link(half_map_2, os.path.join(output_folder, f"{emdid}_half2.map"))
    if angpix is None:
        with mrcfile.open(half_map, permissive=True, header_only=True) as f:
            angpix = float(f.voxel_size.x)  # type: ignore[union-attr]

    box = _box_size(input_map_path)
    print(f"Soft edge search: widths {START_WIDTH}, {START_WIDTH + STEP_WIDTH} ... px")
    print(f"  Extend inimask: {extend_inimask} px, engine: {engine}, box: {box} px")

    cells = []
    width = START_WIDTH
    while True:
        print(f"\nProcessing soft edge width: {width}")
        cell = _evaluate_width(
            width, input_map_path, half1, emdid, output_folder, extend_inimask, engine,
            n_threads, relion_bin, angpix,
        )  # fmt: skip
        cells.append(cell)
        print(f"  {'PASS' if cell['prfsc_pass'] else 'FAIL'}")
        if cell["prfsc_pass"]:
            stop_reason = "pass"
            break
        if cell["touches_box"]:
            stop_reason = "box_edge"
            break
        if width + STEP_WIDTH > box:
            stop_reason = "box_size"
            break
        width += STEP_WIDTH

    return _summarize(
        cells, stop_reason, emdid, output_folder, extend_inimask, engine
    )


def _summarize(cells, stop_reason, emdid, output_folder, extend_inimask, engine):
    """Keep the chosen soft mask only, write the per-width table and the summary."""
    chosen = cells[-1]
    chosen_width = chosen["width"]
    chosen_mask = os.path.join(
        output_folder,
        f"soft_edge_{chosen_width}",
        f"{emdid}_soft_edge_{chosen_width}.mrc",
    )
    for mrc in glob.glob(os.path.join(output_folder, "soft_edge_*", "*.mrc")):
        if mrc != chosen_mask:
            os.remove(mrc)
    if not os.path.exists(chosen_mask):
        raise RuntimeError(f"Chosen soft mask not found: {chosen_mask}")

    pd.DataFrame(cells).to_csv(
        os.path.join(output_folder, f"{emdid}_all_parameter_results.csv"), index=False
    )
    nan = float("nan")
    summary = {
        "optimal_soft_edge_width": int(chosen_width),
        "prfsc_pass": bool(chosen["prfsc_pass"]),
        "no_passing_width": not chosen["prfsc_pass"],
        "stop_reason": stop_reason,
        "optimal_extend_inimask": extend_inimask,
        "extend_inimask": extend_inimask,
        "engine": engine,
        "optimal_masked_res_0_143": float(chosen.get("masked_res_0_143", nan)),
        "optimal_phase_rand_zero_res": float(chosen.get("phase_rand_zero_res", nan)),
        "optimal_masked_zero_res": float(chosen.get("masked_zero_res", nan)),
        "optimal_corrected_resolution": float(chosen.get("corrected_res_0_143", nan)),
        "optimal_mask_path": chosen_mask,
        "widths_tried": cells,
        "emdid": emdid,
    }
    summary_path = os.path.join(output_folder, f"{emdid}_optimal_mask_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print_report(summary)
    print(f"Summary saved to: {summary_path}")
    return summary


def print_report(summary):
    verdict = "PASS" if summary["prfsc_pass"] else "FAIL (no passing width)"
    print("\nOPTIMAL SOFT EDGE SUMMARY:")
    print(f"   Soft edge width: {summary['optimal_soft_edge_width']} px")
    print(f"   PRFSC criteria: {verdict}")
    print(f"   Stop reason: {summary['stop_reason']}")
    print(f"   Extend inimask: {summary['extend_inimask']} px, engine: {summary['engine']}")
    print("\nWIDTHS TRIED:")
    table = pd.DataFrame(summary["widths_tried"])
    table = table[["width", "prfsc_pass", "touches_box", *RESULT_COLUMNS[1:]]]
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the narrowest soft edge width at which a mask passes the PRFSC criteria."
    )
    parser.add_argument("-i", "--input_map_path", type=str, required=True,
                        help="The input .mrc binary mask map file path")  # fmt: skip
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="The output folder")  # fmt: skip
    parser.add_argument("--half_map", type=str, required=True,
                        help="Path to half map 1 for postprocessing")  # fmt: skip
    parser.add_argument("--half_map_2", type=str, default=None,
                        help="Path to half map 2 (default: derived from half map 1 name)")  # fmt: skip
    parser.add_argument("-e", "--extend_inimask", type=float, default=0,
                        help="Mask extension in pixels before the soft edge, default is 0")  # fmt: skip
    parser.add_argument("--engine", choices=["distance", "relion"], default="distance",
                        help="Soft mask engine: distance transform (default, same values as "
                        "relion_mask_create) or relion_mask_create")  # fmt: skip
    parser.add_argument("--relion_bin", type=str, default=None,
                        help="Directory with the RELION binaries (default: PATH)")  # fmt: skip
    parser.add_argument("--angpix", type=float, default=None,
                        help="Pixel size for relion_postprocess (default: half map header)")  # fmt: skip
    parser.add_argument("-j", "--n_threads", type=int, default=os.cpu_count(),
                        help="The number of threads for relion_mask_create, default is all CPU cores")  # fmt: skip
    args = parser.parse_args()

    try:
        summary = run_soft_edge_search(
            input_map_path=args.input_map_path,
            output_folder=args.output_folder,
            half_map=args.half_map,
            extend_inimask=args.extend_inimask,
            engine=args.engine,
            n_threads=args.n_threads,
            relion_bin=args.relion_bin,
            half_map_2=args.half_map_2,
            angpix=args.angpix,
        )
        print(
            f"\nFINAL RESULT: soft edge width {summary['optimal_soft_edge_width']} px, "
            f"{'pass' if summary['prfsc_pass'] else 'no passing width'} ({summary['stop_reason']})"
        )
    except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as e:
        print(f"Error: {e}")
        exit(1)
