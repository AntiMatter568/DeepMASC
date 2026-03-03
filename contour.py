from pathlib import Path

import mrcfile
import numpy as np
from sklearn import mixture
import os
import subprocess

from skimage.morphology import ball, opening, closing
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from loguru import logger
from utils import run_subprocess_realtime
import shutil
import sys

# change to relative path to this file
CURR_SCIPT_PATH = Path(__file__).absolute().parent
CRYOREAD_PATH = CURR_SCIPT_PATH / "CryoREAD"


def create_spherical_mask(array_shape, radius=95):
    if len(array_shape) != 3:
        raise ValueError("Array must be 3D")

    min_dim = min(array_shape)
    radius = (min_dim * radius / 100) / 2

    z, y, x = np.ogrid[: array_shape[0], : array_shape[1], : array_shape[2]]
    center_z, center_y, center_x = array_shape[0] / 2, array_shape[1] / 2, array_shape[2] / 2

    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)

    return dist_from_center <= radius


def run_cryoREAD(mrc_path, output_folder, batch_size=8, gpu_id=None, contour_level=0.0, debug=False):
    output_folder = str(Path(output_folder).absolute())
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Using output directory: {output_folder}")

    try:
        # Use absolute path to main.py in CryoREAD directory
        cryoread_main_path = CRYOREAD_PATH / "main.py"
        
        # Prepare the command for running CryoREAD
        cmd = [
            sys.executable,
            str(cryoread_main_path),
            "--mode=0",
            f"-F={mrc_path}",
            f"--contour={contour_level}",
            f"--gpu={gpu_id}",
            f"--batch_size={batch_size}",
            f"--prediction_only",
            f"--resolution=2.0",
            f"--output={output_folder}",
        ]

        logger.info("Running CryoREAD command: " + " ".join(cmd))

        # Run the command with CryoREAD directory as working directory
        exit_code = run_subprocess_realtime(cmd)  # No timeout for CryoREAD
        
        if exit_code != 0:
            logger.error(f"CryoREAD process exited with code {exit_code}")
            return False

        logger.info("CryoREAD completed successfully")
        return True  # Return success status

    except Exception as e:
        logger.error(f"Error running CryoREAD: {str(e)}")
        return False


def _load_revised_contour_from_txt(txt_path, aggressive=False):
    """Load revised contour value from gmm_mask output txt file. Returns (revised_contour, mask_percent) or None if parse fails."""
    try:
        with open(txt_path) as f:
            content = f.read()
        revised_contour = None
        revised_contour_agg = None
        mask_percent = None
        for line in content.strip().split("\n"):
            if line.startswith("Revised contour (Aggressive):"):
                revised_contour_agg = float(line.split(":", 1)[1].strip())
            elif line.startswith("Revised contour:"):
                revised_contour = float(line.split(":", 1)[1].strip())
            elif line.startswith("Masked percentage:"):
                mask_percent = float(line.split(":", 1)[1].strip())
        if revised_contour_agg is None:
            revised_contour_agg = revised_contour
        contour = revised_contour_agg if aggressive else revised_contour
        if contour is not None and mask_percent is not None:
            return contour, mask_percent
    except (OSError, ValueError) as e:
        logger.debug(f"Could not load revised contour from {txt_path}: {e}")
    return None


def save_mrc(orig_map_path, data, out_path):
    with mrcfile.open(orig_map_path, permissive=True) as orig_map:
        with mrcfile.new(out_path, data=data.astype(np.float32), overwrite=True) as mrc:
            mrc.voxel_size = orig_map.voxel_size
            mrc.header.nxstart = orig_map.header.nxstart
            mrc.header.nystart = orig_map.header.nystart
            mrc.header.nzstart = orig_map.header.nzstart
            mrc.header.origin = orig_map.header.origin
            mrc.header.mapc = orig_map.header.mapc
            mrc.header.mapr = orig_map.header.mapr
            mrc.header.maps = orig_map.header.maps
            mrc.update_header_stats()
            mrc.update_header_from_data()
            mrc.flush()


def gen_features(map_array):
    non_zero_data = map_array[np.nonzero(map_array)]
    data_normalized = (map_array - map_array.min()) * 2 / (map_array.max() - map_array.min()) - 1
    local_grad_norm = rank.gradient(img_as_ubyte(data_normalized), ball(3))
    local_grad_norm = local_grad_norm[np.nonzero(map_array)]
    local_grad_norm = (local_grad_norm - local_grad_norm.min()) / (local_grad_norm.max() - local_grad_norm.min())
    non_zero_data_normalized = (non_zero_data - non_zero_data.min()) / (non_zero_data.max() - non_zero_data.min())

    # stack the flattened data and gradient
    local_grad_norm = np.reshape(local_grad_norm, (-1, 1))
    non_zero_data_normalized = np.reshape(non_zero_data_normalized, (-1, 1))
    features = np.hstack((non_zero_data_normalized, local_grad_norm))

    return features


def gmm_mask(
    input_map_path,
    output_folder,
    num_components=2,
    use_grad=False,
    n_init=1,
    plot_all=False,
    morph_radius=3,
    mask_diameter=95,
    aggressive=False,
):
    logger.info(f"Input map path: {input_map_path}")
    logger.info(f"Output folder: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    logger.info("Opening map file")

    with mrcfile.open(input_map_path, permissive=True) as mrc:
        map_data = mrc.data.copy()

    # generate a spherical mask to mitigate artifacts from padding skip
    if mask_diameter != 0:
        logger.info(f"Applying spherical mask with a diameter of {mask_diameter} % of smallest dimension")
        sphere_mask = create_spherical_mask(map_data.shape, radius=mask_diameter)
        map_data = np.where(sphere_mask, map_data, 0)

    logger.info(f"Input map shape: {map_data.shape}")

    # apply spherical mask
    non_zero_data = map_data[np.nonzero(map_data)]

    data_normalized = (map_data - map_data.min()) * 2 / (map_data.max() - map_data.min()) - 1

    logger.info(f"Non-zero data shape: {non_zero_data.shape}")

    # Zooming to handling large maps
    if len(non_zero_data) >= 5e6:
        logger.warning("Map is too large, resizing...")

        # resample
        zoom_factor = (2e6 / len(non_zero_data)) ** (1 / 3)
        logger.info(f"Resample with zoom factor: {zoom_factor}")

        map_data_zoomed = zoom(map_data, zoom_factor, order=3, mode="grid-constant", grid_mode=False)
        data_normalized_zoomed = (map_data_zoomed - map_data_zoomed.min()) * 2 / (map_data_zoomed.max() - map_data_zoomed.min()) - 1
        non_zero_data_zoomed = map_data_zoomed[np.nonzero(map_data_zoomed)]

        logger.info(f"Shape after resample: {data_normalized_zoomed.shape}")

        logger.info("Calculating gradient")
        local_grad_norm_zoomed = rank.gradient(img_as_ubyte(data_normalized_zoomed), ball(3))
        local_grad_norm_zoomed = local_grad_norm_zoomed[np.nonzero(map_data_zoomed)]
        local_grad_norm_zoomed = (local_grad_norm_zoomed - local_grad_norm_zoomed.min()) / (
            local_grad_norm_zoomed.max() - local_grad_norm_zoomed.min()
        )

        non_zero_data_normalized_zoomed = (non_zero_data_zoomed - non_zero_data_zoomed.min()) / (
            non_zero_data_zoomed.max() - non_zero_data_zoomed.min()
        )

        local_grad_norm_zoomed = np.reshape(local_grad_norm_zoomed, (-1, 1))
        non_zero_data_normalized_zoomed = np.reshape(non_zero_data_normalized_zoomed, (-1, 1))
        data_zoomed = np.hstack((non_zero_data_normalized_zoomed, local_grad_norm_zoomed))

    # calculate Gaussian gradient norm
    local_grad_norm = rank.gradient(img_as_ubyte(data_normalized), ball(3))
    local_grad_norm = local_grad_norm[np.nonzero(map_data)]

    # min-max normalization
    local_grad_norm = (local_grad_norm - local_grad_norm.min()) / (local_grad_norm.max() - local_grad_norm.min())
    non_zero_data_normalized = (non_zero_data - non_zero_data.min()) / (non_zero_data.max() - non_zero_data.min())

    # stack the flattened data and gradient
    local_grad_norm = np.reshape(local_grad_norm, (-1, 1))
    non_zero_data_normalized = np.reshape(non_zero_data_normalized, (-1, 1))
    data = np.hstack((non_zero_data_normalized, local_grad_norm))

    logger.info("Fitting GMM")

    # fit the GMM
    # Use BayesianGaussianMixture for better regularization and uncertainty handling
    # Note: BayesianGaussianMixture can collapse components via its Dirichlet prior,
    # so we need to handle cases where fewer components than requested are returned
    # If components collapse, automatically retry with one extra component
    
    # Use gradient as feature or not
    if use_grad:
        data_to_fit = data_zoomed if len(non_zero_data) >= 5e6 else data
        data_to_predict = data
    else:
        data_to_fit = non_zero_data_normalized_zoomed if len(non_zero_data) >= 5e6 else non_zero_data_normalized
        data_to_predict = non_zero_data_normalized
    
    # Adaptive component selection: retry with extra component if collapse occurs
    effective_num_components = num_components
    max_retries = 1  # Only retry once with one extra component
    retry_count = 0
    
    while retry_count <= max_retries:
        logger.info(f"Fitting feature shape: {data_to_fit.shape} with {effective_num_components} components")
        g = mixture.BayesianGaussianMixture(n_components=effective_num_components, max_iter=500, n_init=n_init, tol=1e-2)
        g.fit(data_to_fit)
        
        # Check if Bayesian GMM collapsed any components
        n_components_found = len(np.unique(g.predict(data_to_fit)))
        if n_components_found < effective_num_components:
            logger.warning(f"Bayesian GMM collapsed components: requested {effective_num_components}, found {n_components_found}")
            if retry_count < max_retries:
                effective_num_components += 1
                retry_count += 1
                logger.info(f"Retrying with {effective_num_components} components")
                continue
            else:
                logger.warning(f"Components still collapsed after retry. Using {n_components_found} components found.")
        else:
            logger.info(f"Successfully fitted {n_components_found} components")

        # Predict on full-res data (may differ from fit data when map was resized)
        logger.info(f"Predicting, feature shape: {data_to_predict.shape}")
        preds = g.predict(data_to_predict)
        ind_noise = np.argmin(np.abs(g.means_[:, 0].flatten()))
        noise_comp = map_data[np.nonzero(map_data)][preds == ind_noise]

        # Empty noise component can occur when full-res prediction differs from fit data
        if noise_comp.size == 0:
            if retry_count < max_retries:
                logger.warning("Bayesian GMM noise component is empty on full-res prediction; retrying with extra component")
                effective_num_components += 1
                retry_count += 1
                logger.info(f"Retrying with {effective_num_components} components")
                continue
            else:
                logger.warning("Bayesian GMM noise component is empty; using minimum non-zero density as revised contour")
                revised_contour = np.min(map_data[np.nonzero(map_data)])
                break
        else:
            break

    # plot the histogram
    if plot_all:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        all_datas = []
        unique_preds = np.unique(preds)
        for pred in unique_preds:
            mask = np.zeros_like(map_data)
            mask[np.nonzero(map_data)] = preds == pred
            masked_map_data = map_data * mask
            new_data_non_zero = masked_map_data[np.nonzero(masked_map_data)]
            all_datas.append(new_data_non_zero.flatten())
            mean = np.mean(new_data_non_zero)
            ax.axvline(mean, linestyle="--", color="k", label=f"Mean_{pred}")
        # Use actual number of components found for labels
        labels = [f"Component {i}" for i in range(len(unique_preds))]
        ax.hist(all_datas, alpha=0.5, bins=256, density=True, log=True, label=labels, stacked=True)
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.set_xlabel("Map Density Value")
        ax.set_ylabel("Density (log scale)")
        ax.set_title("Stacked Histogram by Component")
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, Path(input_map_path).stem + "_hist_by_components.png"))

    # choose ind that is closest to 0, and ind that has the highest mean
    ind_noise = np.argmin(np.abs(g.means_[:, 0].flatten()))

    if use_grad and g.means_.shape[1] > 1:
        logger.debug(f"Means: {g.means_.shape}, density={g.means_[:, 0]}, gradient={g.means_[:, 1]}")
    else:
        logger.debug(f"Means: {g.means_.shape}, density={g.means_[:, 0]}")

    # generate a mask to keep only the component without the noise
    mask = np.zeros_like(map_data)
    mask[np.nonzero(map_data)] = preds != ind_noise

    noise_comp = map_data[np.nonzero(map_data)][preds == ind_noise]
    if noise_comp.size == 0:
        logger.warning("Bayesian GMM noise component is empty; using minimum non-zero density as revised contour")
        revised_contour = np.min(map_data[np.nonzero(map_data)])
    else:
        revised_contour = np.max(noise_comp)

    prot_comp = map_data[np.nonzero(map_data)][preds != ind_noise]

    logger.info(f"Revised contour: {revised_contour}")
    logger.info(f"Remaining mask region size in voxels: {np.count_nonzero(mask)}")

    # use closing then opening to remove small holes
    mask = closing(mask.astype(bool), ball(morph_radius))
    mask = opening(mask.astype(bool), ball(morph_radius))
    masked_map_data = map_data * mask
    new_data_non_zero = masked_map_data[np.nonzero(masked_map_data)]

    # calculate new gradient norm
    new_fit_data = gen_features(masked_map_data)
    logger.info(f"Fitting feature shape: {new_fit_data.shape}")

    # fit the GMM again on the new data (aggressive masking)
    # Use adaptive component selection here too
    effective_num_components_2 = 2
    max_retries_2 = 1
    retry_count_2 = 0
    
    while retry_count_2 <= max_retries_2:
        logger.info(f"Fitting second GMM with {effective_num_components_2} components")
        g2 = mixture.BayesianGaussianMixture(n_components=effective_num_components_2, max_iter=500, n_init=n_init, tol=1e-2)
        g2.fit(new_fit_data)
        
        # Check if components collapsed
        n_components_found_2 = len(np.unique(g2.predict(new_fit_data)))
        if n_components_found_2 < effective_num_components_2:
            logger.warning(f"Second Bayesian GMM collapsed components: requested {effective_num_components_2}, found {n_components_found_2}")
            if retry_count_2 < max_retries_2:
                effective_num_components_2 += 1
                retry_count_2 += 1
                logger.info(f"Retrying second GMM with {effective_num_components_2} components")
                continue
            else:
                logger.warning(f"Second GMM components still collapsed after retry. Using {n_components_found_2} components found.")
        else:
            logger.info(f"Successfully fitted second GMM with {n_components_found_2} components")
        break

    # predict the new data
    new_preds = g2.predict(new_fit_data)
    ind_noise_second = np.argmin(g2.covariances_[:, 0, 0].flatten())
    noise_comp_2 = masked_map_data[np.nonzero(masked_map_data)][new_preds == ind_noise_second]
    prot_comp_2 = masked_map_data[np.nonzero(masked_map_data)][new_preds != ind_noise_second]
    if noise_comp_2.size == 0:
        logger.warning("Second Bayesian GMM noise component is empty; using revised_contour from first GMM")
        revised_contour_agg = revised_contour
    else:
        revised_contour_agg = np.max(noise_comp_2)

    logger.info(f"Revised contour (Aggressive): {revised_contour_agg:.3f}")

    mask_percent = np.count_nonzero(masked_map_data > 1e-8) / np.count_nonzero(map_data > 1e-8)

    # save the new data
    # save_mrc(input_map_path, masked_map_data, os.path.join(output_folder, Path(input_map_path).stem + "_mask.mrc"))
    agg_mask = np.zeros_like(map_data)
    agg_mask[np.nonzero(masked_map_data)] = new_preds != ind_noise_second

    # Apply the same morphological operations to the aggressive mask
    agg_mask = closing(agg_mask.astype(bool), ball(morph_radius))
    agg_mask = opening(agg_mask.astype(bool), ball(morph_radius))

    save_mrc(input_map_path, mask, os.path.join(output_folder, "prot_mask.mrc"))
    save_mrc(input_map_path, agg_mask, os.path.join(output_folder, "prot_mask_aggressive.mrc"))

    # plot the histogram
    plt.style.use("seaborn-v0_8-whitegrid")  # Use modern seaborn style
    fig, ax = plt.subplots(figsize=(12, 5))  # Larger figure size

    # Plot histograms with better colors
    ax.hist(non_zero_data.flatten(), bins=256, density=False, log=True, color="#3498db", alpha=0.7, label="Original")
    ax.hist(new_data_non_zero.flatten(), bins=256, density=False, log=True, color="#e74c3c", alpha=0.7, label="Masked")

    # Add contour lines with better visibility
    ax.axvline(revised_contour, label="Revised Contour (Conservative)", linestyle="dashed", color="#2c3e50", linewidth=2)
    ax.axvline(revised_contour_agg, label="Revised Contour (Aggressive)", linestyle="dotted", color="#27ae60", linewidth=2)

    # Improve labels and formatting
    ax.set_xlabel("Density Value", fontsize=12)
    ax.set_ylabel("Frequency (log scale)", fontsize=12)
    ax.set_title(f"Density Distribution: {Path(input_map_path).stem}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=10)

    # Improve legend
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Tight layout and save with higher DPI
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, Path(input_map_path).stem + "_hist_overall.png"), dpi=300)
    plt.close(fig)  # Close the figure to free memory
    out_txt = os.path.join(output_folder, Path(input_map_path).stem + "_revised_contour.txt")

    with open(out_txt, "w") as f:
        f.write(f"Revised contour: {revised_contour}\n")
        f.write(f"Revised contour (Aggressive): {revised_contour_agg}\n")
        f.write(f"Masked percentage: {mask_percent}\n")

    # return revised contour level and mask percent
    if aggressive:
        return revised_contour_agg, mask_percent
    return revised_contour, mask_percent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_map_path", type=str, required=True, help="The input map path")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="The output folder")
    parser.add_argument("-g", "--gpu_id", type=str, required=False, help="The gpu id for cryoREAD prediction (required if -r is specified)")
    parser.add_argument("-p", "--plot_all", action="store_true", help="Draw a plot for each of the components")
    parser.add_argument("-n", "--num_components", type=int, default=2, help="Number of components for mixture model, default is 2")
    parser.add_argument("-r", "--refinement_mask", action="store_true", help="Generate more fine-grained mask for refinement")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="The batch size for cryoREAD prediction, default is 8")
    parser.add_argument("-m", "--morph_radius", type=int, default=3, help="The radius for morphological operations (opening, closing), default is 3")
    parser.add_argument(
        "-d",
        "--mask_diameter",
        type=int,
        default=95,
        choices=range(0, 101),
        help="The diameter of the mask in percentage to the shortest dimension of the map (from 0 to 100), set to 0 to disable, default is 95",
    )
    parser.add_argument("-a", "--aggressive", action="store_true", help="Use more aggressive mask cutoff when using GMM mask")
    parser.add_argument("-c", "--cutoff_prob", type=float, default=0.3, help="The cutoff probability for the mask if using CryoREAD mask, default is 0.3")
    parser.add_argument("--resume", action="store_true", help="Resume from existing GMM/CryoREAD outputs; skip steps whose outputs already exist")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Validate that gpu_id is provided when refinement_mask is enabled
    if args.refinement_mask and args.gpu_id is None:
        parser.error("-g/--gpu_id is required when -r/--refinement_mask is specified")

    final_out_mask_path = os.path.join(args.output_folder, "prot_mask_final.mrc")
    
    input_map_path = os.path.abspath(args.input_map_path)
    output_folder = os.path.abspath(args.output_folder)

    # Resume: skip GMM if outputs already exist
    revised_contour_txt = os.path.join(output_folder, Path(input_map_path).stem + "_revised_contour.txt")
    gmm_outputs_exist = (
        os.path.exists(os.path.join(output_folder, "prot_mask.mrc"))
        and os.path.exists(os.path.join(output_folder, "prot_mask_aggressive.mrc"))
        and os.path.exists(revised_contour_txt)
    )
    if args.resume and gmm_outputs_exist:
        loaded = _load_revised_contour_from_txt(revised_contour_txt, aggressive=args.aggressive)
        if loaded is not None:
            revised_contour, mask_percent = loaded
            logger.info(f"Resuming: loaded revised contour from {revised_contour_txt} (contour={revised_contour:.4f})")
        else:
            revised_contour, mask_percent = gmm_mask(
                input_map_path=input_map_path,
                output_folder=output_folder,
                num_components=args.num_components,
                use_grad=True,
                n_init=3,
                plot_all=args.plot_all,
                morph_radius=args.morph_radius,
                mask_diameter=args.mask_diameter,
                aggressive=args.aggressive,
            )
    else:
        revised_contour, mask_percent = gmm_mask(
            input_map_path=input_map_path,
            output_folder=output_folder,
            num_components=args.num_components,
            use_grad=True,
            n_init=3,
            plot_all=args.plot_all,
            morph_radius=args.morph_radius,
            mask_diameter=args.mask_diameter,
            aggressive=args.aggressive,
        )

    if args.refinement_mask:
        cryoread_prob_path = os.path.join(output_folder, "2nd_stage_detection", "chain_protein_prob.mrc")
        if args.resume and os.path.exists(cryoread_prob_path):
            logger.info(f"Resuming: skipping CryoREAD, using existing {cryoread_prob_path}")
        else:
            if not run_cryoREAD(
                mrc_path=input_map_path,
                output_folder=output_folder,
                batch_size=args.batch_size,
                gpu_id=args.gpu_id,
                contour_level=revised_contour,
                debug=args.debug,
            ):
                logger.error("CryoREAD failed to run, please check the log file")
                exit(1)

        # load the mask
        with mrcfile.open(os.path.join(output_folder, "2nd_stage_detection", "chain_protein_prob.mrc"), permissive=True) as mrc:
            protein_prob = mrc.data.copy()
            # binarize the protein probability map
            protein_prob = protein_prob > args.cutoff_prob

        # make morphological operations on the mask
        mask = opening(protein_prob.astype(bool), ball(args.morph_radius))
        mask = closing(mask.astype(bool), ball(args.morph_radius))
        
        # save the mask
        save_mrc(os.path.join(output_folder, "input.mrc"), mask, final_out_mask_path)

    else:
        if args.aggressive:
            shutil.copy(os.path.join(output_folder, "prot_mask_aggressive.mrc"), final_out_mask_path)
        else:
            shutil.copy(os.path.join(output_folder, "prot_mask.mrc"), final_out_mask_path)

    # Resample the final mask to match the original input map when refinement_mask is enabled
    if args.refinement_mask:
        final_out_mask_path_resampled = os.path.join(output_folder, "prot_mask_final_resampled.mrc")
        if args.resume and os.path.exists(final_out_mask_path_resampled) and not os.path.islink(final_out_mask_path_resampled):
            logger.info(f"Resuming: skipping Chimera resampling, using existing {final_out_mask_path_resampled}")
        else:
            resample_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resample_chimera.py")
            cmd = ["chimera", "--nogui", resample_script_path, final_out_mask_path, input_map_path, final_out_mask_path_resampled]
            logger.info(f"Running chimera resampling command: {' '.join(cmd)}")
            try:
                exit_code = run_subprocess_realtime(cmd, timeout=300)  # 5 minutes timeout for Chimera
            except subprocess.TimeoutExpired:
                logger.error("Chimera resampling timed out after 300 seconds")
                raise ValueError("# Logical Error: Chimera resampling timed out")
            
            if exit_code != 0:
                logger.error(f"Chimera failed with exit code {exit_code}")
                raise ValueError(f"# Logical Error: Chimera failed with exit code {exit_code}")
            else:
                logger.info("Resampling completed successfully")
    else:
        # Create a symlink to indicate no resampling was done
        final_out_mask_path_resampled = os.path.join(output_folder, "prot_mask_final_resampled.mrc")
        if os.path.exists(final_out_mask_path_resampled):
            os.remove(final_out_mask_path_resampled)
        os.symlink(os.path.abspath(final_out_mask_path), final_out_mask_path_resampled)
