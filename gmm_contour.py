from pathlib import Path

import mrcfile
import numpy as np
from sklearn import mixture
import os

from skimage.morphology import ball, opening
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy.ndimage import zoom

import matplotlib.pyplot as plt


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


def gmm_mask(input_map_path, output_folder, num_components=2, use_grad=False, n_init=1, plot_all=False):
    print("input_map_path", input_map_path)
    print("output_folder", output_folder)

    os.makedirs(output_folder, exist_ok=True)

    print("Opening map file")

    with mrcfile.open(input_map_path, permissive=True) as mrc:
        map_data = mrc.data.copy()

    print("Input map shape:", map_data.shape)

    non_zero_data = map_data[np.nonzero(map_data)]

    data_normalized = (map_data - map_data.min()) * 2 / (map_data.max() - map_data.min()) - 1

    print("Non-zero data shape", non_zero_data.shape)

    # Zooming to handling large maps
    if len(non_zero_data) >= 5e6:
        print("Map is too large, resizing...")

        # resample
        zoom_factor = (2e6 / len(non_zero_data)) ** (1 / 3)
        print("Resample with zoom factor:", zoom_factor)

        map_data_zoomed = zoom(map_data, zoom_factor, order=3, mode="grid-constant", grid_mode=False)
        data_normalized_zoomed = (map_data_zoomed - map_data_zoomed.min()) * 2 / (map_data_zoomed.max() - map_data_zoomed.min()) - 1
        non_zero_data_zoomed = map_data_zoomed[np.nonzero(map_data_zoomed)]

        print("Shape after resample:", data_normalized_zoomed.shape)

        print("Calculating gradient")
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
        # print(non_zero_data_normalized_zoomed.shape, local_grad_norm_zoomed.shape)
        data_zoomed = np.hstack((non_zero_data_normalized_zoomed, local_grad_norm_zoomed))

    # calculate guassian gradient norm
    local_grad_norm = rank.gradient(img_as_ubyte(data_normalized), ball(3))
    local_grad_norm = local_grad_norm[np.nonzero(map_data)]

    # min-max normalization
    local_grad_norm = (local_grad_norm - local_grad_norm.min()) / (local_grad_norm.max() - local_grad_norm.min())
    non_zero_data_normalized = (non_zero_data - non_zero_data.min()) / (non_zero_data.max() - non_zero_data.min())

    # stack the flattened data and gradient
    local_grad_norm = np.reshape(local_grad_norm, (-1, 1))
    non_zero_data_normalized = np.reshape(non_zero_data_normalized, (-1, 1))
    data = np.hstack((non_zero_data_normalized, local_grad_norm))

    print("Fitting GMM")

    # fit the GMM
    g = mixture.BayesianGaussianMixture(n_components=num_components, max_iter=500, n_init=n_init, tol=1e-2)

    if use_grad:
        data_to_fit = data_zoomed if len(non_zero_data) >= 5e6 else data
    else:
        data_to_fit = non_zero_data_normalized_zoomed if len(non_zero_data) >= 5e6 else non_zero_data_normalized
    print("Fitting feature shape:", data_to_fit.shape)
    g.fit(data_to_fit)
    print("Predicting, feature shape:", data.shape)
    preds = g.predict(data)

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
        # print("Saving figure to", os.path.join(output_folder, "hist_by_component.png"))
        fig.savefig(os.path.join(output_folder, Path(input_map_path).stem + "_hist_by_components.png"))

    # choose ind that is closest to 0, and ind that has the highest mean
    ind_noise = np.argmin(np.abs(g.means_[:, 0].flatten()))
    # ind_max = np.argmax(g.covariances_[:, 0, 0].flatten())

    print("Means: ", g.means_.shape, g.means_[:, 0], g.means_[:, 1])

    # print("Variances: ", g.covariances_.shape, g.covariances_[ind_noise, 0, 0])
    # print("Std: ", np.sqrt(g.covariances_[ind_noise, 0, 0]))

    # generate a mask to keep only the component without the noise
    mask = np.zeros_like(map_data)
    mask[np.nonzero(map_data)] = preds != ind_noise

    noise_comp = map_data[np.nonzero(map_data)][preds == ind_noise]
    revised_contour = np.max(noise_comp)

    prot_comp = map_data[np.nonzero(map_data)][preds != ind_noise]

    print("Revised contour:", revised_contour)
    print("Remaining mask region size in voxels:", np.count_nonzero(mask))

    # use opening to remove small holes
    mask = opening(mask.astype(bool), ball(3))
    masked_map_data = map_data * mask
    new_data_non_zero = masked_map_data[np.nonzero(masked_map_data)]

    # calculate new gradient norm
    new_fit_data = gen_features(masked_map_data)
    print("Fitting feature shape:", new_fit_data.shape)

    # fit the GMM again on the new data
    g2 = mixture.BayesianGaussianMixture(n_components=2, max_iter=500, n_init=n_init, tol=1e-2)
    g2.fit(new_fit_data)

    # predict the new data
    new_preds = g2.predict(new_fit_data)
    # ind_noise_second = np.argmin(np.abs(g2.means_[:, 0].flatten()))
    ind_noise_second = np.argmin(g2.covariances_[:, 0, 0].flatten())
    noise_comp_2 = masked_map_data[np.nonzero(masked_map_data)][new_preds == ind_noise_second]
    prot_comp_2 = masked_map_data[np.nonzero(masked_map_data)][new_preds != ind_noise_second]
    # revised_contour_2 = np.median(prot_comp_2)
    revised_contour_2 = np.max(noise_comp_2)

    print("Revised contour (high):", revised_contour_2)

    # save the new data
    save_mrc(input_map_path, masked_map_data, os.path.join(output_folder, Path(input_map_path).stem + "_mask.mrc"))

    mask_percent = np.count_nonzero(masked_map_data > 1e-8) / np.count_nonzero(map_data > 1e-8)

    # plot the histogram
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.hist(non_zero_data.flatten(), alpha=0.5, bins=256, density=False, log=True, label="Original")
    ax.hist(new_data_non_zero.flatten(), alpha=0.5, bins=256, density=False, log=True, label="Masked")
    ax.axvline(revised_contour, label="Revised Contour")
    ax.axvline(revised_contour_2, label="Revised Contour (High)", linestyle="dashed")
    ax.legend()
    plt.title(input_map_path)
    plt.savefig(os.path.join(output_folder, Path(input_map_path).stem + "_hist_overall.png"))

    out_txt = os.path.join(output_folder, Path(input_map_path).stem + "_revised_contour.txt")

    with open(out_txt, "w") as f:
        f.write(f"Revised contour: {revised_contour}\n")
        f.write(f"Revised contour (high): {revised_contour_2}\n")
        f.write(f"Masked percentage: {mask_percent}\n")

    # return revised contour level and mask percent
    return revised_contour, mask_percent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_map_path", type=str, default=None)
    parser.add_argument("-o", "--output_folder", type=str, default=None)
    parser.add_argument("-p", "--plot_all", action="store_true")
    parser.add_argument("-n", "--num_components", type=int, default=2)
    args = parser.parse_args()
    revised_contour, mask_percent = gmm_mask(
        input_map_path=args.input_map_path,
        output_folder=args.output_folder,
        num_components=args.num_components,
        use_grad=True,
        n_init=3,
        plot_all=args.plot_all,
    )
