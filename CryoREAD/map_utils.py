import mrcfile
import numpy as np


def calc_map_ccc(input_mrc, input_pred, center=True, overlap_only=False):
    """
    Calculate the Concordance Correlation Coefficient (CCC) and overlap percentage of two input MRC files.

    Parameters:
    input_mrc (str): Path to the MRC file.
    input_pred (str): Path to the prediction MRC file.
    center (bool, optional): If True, center the data. Defaults to True.

    Returns:
    float: The calculated CCC.
    float: The overlap percentage.
    """
    # Open the MRC files and copy their data
    with mrcfile.open(input_mrc) as mrc:
        mrc_data = mrc.data.copy()
    with mrcfile.open(input_pred) as mrc:
        pred_data = mrc.data.copy()

    # mrc_data = np.where(mrc_data > 1e-8, mrc_data, 0.0)
    # pred_data = np.where(pred_data > 1e-8, pred_data, 0.0)

    # Determine the minimum count of non-zero values
    min_count = np.min([np.count_nonzero(mrc_data), np.count_nonzero(pred_data)])

    # Calculate the overlap of non-zero values
    overlap = mrc_data * pred_data > 0.0

    if overlap_only:
        mrc_data = mrc_data[overlap]
        pred_data = pred_data[overlap]

    # Center the data if specified
    if center:
        mrc_data = mrc_data - np.mean(mrc_data)
        pred_data = pred_data - np.mean(pred_data)

    # Calculate the overlap percentage
    overlap_percent = np.sum(overlap) / min_count

    # Calculate the CCC
    ccc = np.sum(mrc_data * pred_data) / np.sqrt(np.sum(mrc_data**2) * np.sum(pred_data**2))

    return ccc, overlap_percent


"""Compute FSC between two volumes, adapted from cryodrgn"""
import numpy as np
import torch
from torch.fft import fftshift, ifftshift, fft2, fftn, ifftn


def normalize(img, mean=0, std=None, std_n=None):
    if std is None:
        # Since std is a memory consuming process, use the first std_n samples for std determination
        std = torch.std(img[:std_n, ...])

    # logger.info(f"Normalized by {mean} +/- {std}")
    return (img - mean) / std


def fft2_center(img):
    return fftshift(fft2(fftshift(img, dim=(-1, -2))), dim=(-1, -2))


def fftn_center(img):
    return fftshift(fftn(fftshift(img)))


def ifftn_center(img):
    if isinstance(img, np.ndarray):
        # Note: We can't just typecast a complex ndarray using torch.Tensor(array) !
        img = torch.complex(torch.Tensor(img.real), torch.Tensor(img.imag))
    x = ifftshift(img)
    y = ifftn(x)
    z = ifftshift(y)
    return z


def ht2_center(img):
    _img = fft2_center(img)
    return _img.real - _img.imag


def htn_center(img):
    _img = fftshift(fftn(fftshift(img)))
    return _img.real - _img.imag


def iht2_center(img):
    img = fft2_center(img)
    img /= img.shape[-1] * img.shape[-2]
    return img.real - img.imag


def ihtn_center(img):
    img = fftshift(img)
    img = fftn(img)
    img = fftshift(img)
    img /= torch.prod(torch.tensor(img.shape, device=img.device))
    return img.real - img.imag


def symmetrize_ht(ht):
    if ht.ndim == 2:
        ht = ht[np.newaxis, ...]
    assert ht.ndim == 3
    n = ht.shape[0]

    D = ht.shape[-1]
    sym_ht = torch.empty((n, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
    sym_ht[:, 0:-1, 0:-1] = ht

    assert D % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0, :]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]  # last corner is first corner

    if n == 1:
        sym_ht = sym_ht[0, ...]

    return sym_ht


def calculate_fsc(vol1_f, vol2_f, output_f, Apix=1.0, plot=True):

    import mrcfile

    with mrcfile.open(vol1_f, permissive=True) as v1:
        vol1 = v1.data.copy()
    with mrcfile.open(vol2_f, permissive=True) as v2:
        vol2 = v2.data.copy()

    assert vol1.shape == vol2.shape

    # pad if non-cubic
    padding_xyz = np.max(vol1.shape) - vol1.shape

    vol1 = np.pad(vol1, ((0, padding_xyz[0]), (0, padding_xyz[1]), (0, padding_xyz[2])), mode="constant")
    vol2 = np.pad(vol2, ((0, padding_xyz[0]), (0, padding_xyz[1]), (0, padding_xyz[2])), mode="constant")

    if vol1.shape[0] % 2 != 0:
        vol1 = np.pad(vol1, ((0, 1), (0, 1), (0, 1)), mode="constant")
        vol2 = np.pad(vol2, ((0, 1), (0, 1), (0, 1)), mode="constant")

    vol1 = torch.from_numpy(vol1).to(torch.float32)
    vol2 = torch.from_numpy(vol2).to(torch.float32)

    D = vol1.shape[0]
    x = np.arange(-D // 2, D // 2)
    x2, x1, x0 = np.meshgrid(x, x, x, indexing="ij")
    coords = np.stack((x0, x1, x2), -1)
    r = (coords**2).sum(-1) ** 0.5

    assert r[D // 2, D // 2, D // 2] == 0.0, r[D // 2, D // 2, D // 2]

    vol1 = fftn_center(vol1)
    vol2 = fftn_center(vol2)

    prev_mask = np.zeros((D, D, D), dtype=bool)
    fsc = [1.0]
    for i in range(1, D // 2):
        mask = r < i
        shell = np.where(mask & np.logical_not(prev_mask))
        v1 = vol1[shell]
        v2 = vol2[shell]
        p = np.vdot(v1, v2) / (np.vdot(v1, v1) * np.vdot(v2, v2)) ** 0.5
        fsc.append(float(p.real))
        prev_mask = mask
    fsc = np.asarray(fsc)
    x = np.arange(D // 2) / D

    res = np.stack((x, fsc), 1)
    if output_f:
        np.savetxt(output_f, res)
    else:
        # logger.info(res)
        pass

    w = np.where(fsc < 0.5)
    cutoff_05 = 1 / x[w[0][0]] * Apix if len(w) > 0 and len(w[0]) > 0 else None

    if cutoff_05 is None:
        cutoff_05 = 0.0
    
    w = np.where(fsc < 0.143)
    cutoff_0143 = 1 / x[w[0][0]] * Apix if len(w) > 0 and len(w[0]) > 0 else None

    if cutoff_0143 is None:
        cutoff_0143 = 0.0

    # Visualization
    if plot:
        import matplotlib.pyplot as plt
        
        # Convert spatial frequency to resolution in Angstroms
        resolution = 1 / (x * Apix + 1e-10)  # Add small epsilon to avoid division by zero
        resolution[0] = np.inf  # Set DC component to infinity
        
        plt.figure(figsize=(10, 6))
        plt.plot(x[1:], fsc[1:], 'b-', linewidth=2, label='FSC')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='FSC = 0.5')
        plt.axhline(y=0.143, color='orange', linestyle='--', alpha=0.7, label='FSC = 0.143')
        
        # Mark resolution cutoffs
        if cutoff_05 > 0:
            plt.axvline(x=1/(cutoff_05/Apix), color='r', linestyle=':', alpha=0.7)
            plt.text(1/(cutoff_05/Apix), 0.52, f'{cutoff_05:.1f}Å', 
                    rotation=90, verticalalignment='bottom', color='r')
        
        if cutoff_0143 > 0:
            plt.axvline(x=1/(cutoff_0143/Apix), color='orange', linestyle=':', alpha=0.7)
            plt.text(1/(cutoff_0143/Apix), 0.15, f'{cutoff_0143:.1f}Å', 
                    rotation=90, verticalalignment='bottom', color='orange')
        
        plt.xlabel('Spatial Frequency')
        plt.ylabel('Fourier Shell Correlation')
        plt.title('FSC Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, x[-1])
        
        # Add secondary x-axis for resolution
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        
        # Select reasonable tick positions
        freq_ticks = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        freq_ticks = freq_ticks[freq_ticks <= x[-1]]
        res_ticks = 1 / (freq_ticks * Apix)
        
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(freq_ticks)
        ax2.set_xticklabels([f'{r:.1f}' for r in res_ticks])
        ax2.set_xlabel('Resolution (Å)')
        
        plt.tight_layout()
        
        # Generate plot filename based on output_f
        import os
        base_name = os.path.splitext(output_f)[0]
        plot_filename = base_name + "_plot.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"FSC plot saved to: {plot_filename}")
    
    return x, fsc, cutoff_05, cutoff_0143
