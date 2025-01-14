from pathlib import Path
import subprocess

import mrcfile
import numpy as np
from sklearn import mixture
import os

from skimage.morphology import ball, opening, closing
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from loguru import logger

# change to relative path to this file
CURR_SCIPT_PATH = Path(__file__).absolute().parent
CRYOREAD_PATH = CURR_SCIPT_PATH / "CryoREAD"


def create_spherical_mask(array_shape, radius=95):
    if len(array_shape) != 3:
        raise ValueError("Array must be 3D")

    min_dim = min(array_shape)
    radius = (min_dim * radius / 100) / 2

    z, y, x = np.ogrid[:array_shape[0], :array_shape[1], :array_shape[2]]
    center_z, center_y, center_x = array_shape[0] / 2, array_shape[1] / 2, array_shape[2] / 2

    dist_from_center = np.sqrt(
        (x - center_x) ** 2 +
        (y - center_y) ** 2 +
        (z - center_z) ** 2
    )

    return dist_from_center <= radius


def run_cryoREAD(mrc_path, output_folder, batch_size=8, gpu_id=None):
    output_folder = str(Path(output_folder).absolute())
    TEMP_CURR_DIR = os.getcwd()
    os.chdir(CRYOREAD_PATH)

    try:
        cmd = [
            "python",
            "main.py",
            "--mode=0",
            f"-F={mrc_path}",
            "--contour=0",
            f"--gpu={gpu_id}",
            f"--batch_size={batch_size}",
            f"--prediction_only",
            f"--resolution=3.0",
            f"--output={output_folder}",
        ]

        print(" ".join(cmd))
        process = subprocess.run(cmd, shell=False, text=True)
    except:
        print("Error running CryoREAD")
    finally:
        os.chdir(TEMP_CURR_DIR)


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


def gmm_mask(input_map_path, output_folder, num_components=2, use_grad=False, n_init=1, plot_all=False, morph_radius=3, mask_diameter=95, aggressive=False):
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
    g = mixture.BayesianGaussianMixture(n_components=num_components, max_iter=500, n_init=n_init, tol=1e-2)

    # Use gradient as feature or not
    if use_grad:
        data_to_fit = data_zoomed if len(non_zero_data) >= 5e6 else data
    else:
        data_to_fit = non_zero_data_normalized_zoomed if len(non_zero_data) >= 5e6 else non_zero_data_normalized
    logger.info(f"Fitting feature shape: {data_to_fit.shape}")
    g.fit(data_to_fit)
    logger.info(f"Predicting, feature shape: {data.shape}")
    preds = g.predict(data)

    # plot the histogram
    if plot_all:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        all_datas = []
        for pred in np.unique(preds):
            mask = np.zeros_like(map_data)
            mask[np.nonzero(map_data)] = preds == pred
            masked_map_data = map_data * mask
            new_data_non_zero = masked_map_data[np.nonzero(masked_map_data)]
            all_datas.append(new_data_non_zero.flatten())
            mean = np.mean(new_data_non_zero)
            ax.axvline(mean, linestyle="--", color="k", label=f"Mean_{pred}")
        labels = [f"Component {i}" for i in range(num_components)]
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

    logger.info(f"Means: {g.means_.shape}, {g.means_[:, 0]}, {g.means_[:, 1]}")

    # generate a mask to keep only the component without the noise
    mask = np.zeros_like(map_data)
    mask[np.nonzero(map_data)] = preds != ind_noise

    noise_comp = map_data[np.nonzero(map_data)][preds == ind_noise]
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

    # fit the GMM again on the new data
    g2 = mixture.BayesianGaussianMixture(n_components=2, max_iter=500, n_init=n_init, tol=1e-2)
    g2.fit(new_fit_data)

    # predict the new data
    new_preds = g2.predict(new_fit_data)
    ind_noise_second = np.argmin(g2.covariances_[:, 0, 0].flatten())
    noise_comp_2 = masked_map_data[np.nonzero(masked_map_data)][new_preds == ind_noise_second]
    prot_comp_2 = masked_map_data[np.nonzero(masked_map_data)][new_preds != ind_noise_second]
    revised_contour_agg = np.max(noise_comp_2)

    logger.info(f"Revised contour (Aggressive): {revised_contour_agg:.3f}")

    # save the new data
    # save_mrc(input_map_path, masked_map_data, os.path.join(output_folder, Path(input_map_path).stem + "_mask.mrc"))
    if aggressive:
        logger.info("Using more aggressive mask generation")
        agg_mask = np.zeros_like(map_data)
        agg_mask[np.nonzero(masked_map_data)] = new_preds != ind_noise_second
        save_mrc(input_map_path, agg_mask, os.path.join(output_folder, Path(input_map_path).stem + "_mask.mrc"))
    else:
        save_mrc(input_map_path, mask, os.path.join(output_folder, Path(input_map_path).stem + "_mask.mrc"))

    mask_percent = np.count_nonzero(masked_map_data > 1e-8) / np.count_nonzero(map_data > 1e-8)

    # plot the histogram
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.hist(non_zero_data.flatten(), alpha=0.5, bins=256, density=False, log=True, label="Original")
    ax.hist(new_data_non_zero.flatten(), alpha=0.5, bins=256, density=False, log=True, label="Masked")
    ax.axvline(revised_contour, label="Revised Contour (Conservative)", linestyle="dashed")
    ax.axvline(revised_contour_agg, label="Revised Contour (Aggressive)", linestyle="dashed")
    ax.legend()
    plt.title(input_map_path)
    plt.savefig(os.path.join(output_folder, Path(input_map_path).stem + "_hist_overall.png"))
    out_txt = os.path.join(output_folder, Path(input_map_path).stem + "_revised_contour.txt")

    with open(out_txt, "w") as f:
        f.write(f"Revised contour: {revised_contour}\n")
        f.write(f"Revised contour (Aggressive): {revised_contour_agg}\n")
        f.write(f"Masked percentage: {mask_percent}\n")

    # return revised contour level and mask percent
    return revised_contour, mask_percent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_map_path", type=str, required=True, help="The input map path")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="The output folder")
    parser.add_argument("-g", "--gpu_id", type=str, required=True, help="The gpu id for cryoREAD prediction")
    parser.add_argument("-p", "--plot_all", action="store_true", help="Draw a plot for each of the components")
    parser.add_argument("-n", "--num_components", type=int, default=2, help="Number of components for mixture model")
    parser.add_argument("-r", "--refinement_mask", action="store_true", help="Generate more fine-grained mask for refinement")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="The batch size for cryoREAD prediction")
    parser.add_argument("-m", "--morph_radius", type=int, default=3, help="The radius for morphological operations (opening, closing)")
    parser.add_argument("-d", "--mask_diameter", type=int, default=95, choices=range(0,101), help="The diameter of the mask in percentage to the shortest dimension of the map (from 0 to 100), set to 0 to disable")
    parser.add_argument("-a", "--aggressive", action="store_true", help="Use more aggressive mask cutoff when using GMM mask")
    args = parser.parse_args()

    if args.refinement_mask:
        run_cryoREAD(
            mrc_path=args.input_map_path,
            output_folder=args.output_folder,
            batch_size=args.batch_size,
            gpu_id=args.gpu_id,
        )
        final_protein_prob = os.path.join(args.output_folder, "2nd_stage_detection", "chain_protein_prob.mrc")

    else:
        revised_contour, mask_percent = gmm_mask(
            input_map_path=args.input_map_path,
            output_folder=args.output_folder,
            num_components=args.num_components,
            use_grad=True,
            n_init=3,
            plot_all=args.plot_all,
            morph_radius=args.morph_radius,
            mask_diameter=args.mask_diameter,
            aggressive=args.aggressive,
        )
