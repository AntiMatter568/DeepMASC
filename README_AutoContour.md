# AutoContour

AutoContour automatically determines contour levels and generates masks for cryo-EM maps.

## Standalone Usage

<details>

### Arguments

**Required Arguments:**
* `-i, --input_map_path`: Input MRC map file to determine the contour
* `-o, --output_folder`: Output folder to store all the files
* `-g, --gpu_id`: GPU ID to use for CryoREAD prediction. Specifies which GPU device should be used for processing.

**Optional Arguments:**
* `-p, --plot_all`: Plot all components (default: False)
* `-n, --num_components`: Number of components for mixture model (default: 2)
* `-r, --refinement_mask`: Generate more fine-grained mask for refinement (default: False)
* `-b, --batch_size`: Batch size for CryoREAD prediction (default: 8)
* `-m, --morph_radius`: Radius for morphological operations (opening, closing) (default: 3)
* `-d, --mask_diameter`: The diameter of the mask in percentage to the shortest dimension of the map (from 0 to 100), set to 0 to disable (default: 95)
* `-a, --aggressive`: Use more aggressive mask cutoff when using GMM mask (default: False)
* `-c, --cutoff_prob`: The cutoff probability for the mask if using CryoREAD mask (default: 0.3)
* `--debug`: Enable debug mode (default: False)

### Examples

**GMM Auto Contouring for Rough Masking:**

Using the provided example data:
```bash
PYTHONNOUSERSITE=1 python contour.py -i examples/Class3D/job016/run_it025_class001.mrc -o output_autocontour_gmm -g 0 -p
```

Using your own data:
```bash
PYTHONNOUSERSITE=1 python contour.py -i /path/to/your/map.mrc -o output_folder -g 0 -p
```

**CryoREAD Auto Refinement Masking:**

Using the provided example data:
```bash
PYTHONNOUSERSITE=1 python contour.py -i examples/Class3D/job016/run_it025_class001.mrc -o output_autocontour_cryoread -g 0 -p -r -b 16
```

Using your own data:
```bash
PYTHONNOUSERSITE=1 python contour.py -i /path/to/your/map.mrc -o output_folder -g 0 -p -r -b 16
```

</details>

## RELION GUI Integration

<details>

### Files

There is one file associated with RELION integration of AutoContour:

- `gtf_relion4_run_autocontour.py` <- **this is the main file to execute**

### RELION GUI Setup Instructions

**Note:** If you have installed the DeepMASC module system (see `module/README.md`), you can use the convenient wrapper command:
- Use `deepmasc-relion-contour` instead of `python /path/to/gtf_relion4_run_autocontour.py`
- No need to activate conda or specify full paths!

#### Traditional Setup (without module system):

1. From RELION GUI, Choose "External", then in "External Executable" box enter:
   - **External Executable**: `python /path/to/gtf_relion4_run_autocontour.py`

2. In the "Input" tab:
   - In "Input 3D reference" box, select the map file you want to generate a mask for.

3. In the "Params" tab, you can set the following parameters:
   - `gpus`: GPU IDs to use for CryoREAD prediction (required), e.g., `0` or `0,1`
   - `plot_all`: Set to `True` to generate component plots (optional)
   - `num_components`: Number of components for mixture model (optional, default: 2)
   - `refinement_mask`: Set to `True` to use CryoREAD for fine-grained masking (optional)
   - `batch_size`: Batch size for CryoREAD prediction (optional, default: 8)
   - `morph_radius`: Radius for morphological operations (optional, default: 3)
   - `mask_diameter`: Diameter of spherical mask in percentage (optional, default: 95)
   - `aggressive`: Set to `True` for more aggressive masking (optional)
   - `cutoff_prob`: Cutoff probability for CryoREAD mask (optional, default: 0.3)
   - `debug`: Set to `True` to enable debug mode (optional)

4. In the "Running" tab:
   - Set "Number of threads" to 1
   - Adjust your submission to queue settings if using a managed queue system

5. Click the "Run" button to start the job.

6. Once finished, the results will be stored in the output job directory created by RELION, containing all the output files listed in the previous section.

</details>

## Mask Generation Process

<details>

1. (Optional) Apply a spherical mask (diameter controlled by `--mask_diameter`, default: 95% of the map's smallest dimension, set to 0 to disable) to eliminate padding skip artifacts in the corners.
2. Extract all intensity and gradient features from the map where the value is non-zero.
3. Apply a Bayesian GMM (Gaussian Mixture Model) to classify non-zero voxels in the previous step into a specified number of components (controlled by `--num_components`, default: 2) using the features.
4. The component with mean intensity closest to zero is labeled noise and excluded from the mask.
5. (Optional) Morphological operations are applied. Closing (fills holes) followed by opening (removes isolated points) using a spherical kernel with radius controlled by `--morph_radius` (default: 3 pixels, set to 0 to disable) to clean the mask.
6. (Optional) If aggressive masking is enabled (`--aggressive`), a secondary GMM further splits the retained voxels to remove weaker signal regions.
7. The final binary mask is saved as "prot_mask_final.mrc", preserving original voxel dimensions and header metadata. If `--plot_all` is enabled, histograms for each component will be saved.
8. (Optional) When `--refinement_mask` is used, the mask's contour level feeds into CryoREAD's neural network for detailed protein segmentation.

</details>

## List of Output Files

<details>

| File                                | Description                                                                                                                              |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| prot_mask_final.mrc                 | The final binary mask output. This will be either the GMM-based mask or CryoREAD-refined mask depending on your settings                 |
| prot_mask.mrc                       | The conservative GMM-based binary mask, the same as prot_mask_final.mrc if `--aggressive` is not used and not using  `--refinement_mask` |
| prot_mask_aggressive.mrc            | The aggressive GMM-based binary mask, the same as prot_mask_final.mrc if `--aggressive` is used and not using  `--refinement_mask`       |
| {input_name}_hist_overall.png       | Overall density distribution histogram showing original and masked data distributions                                                    |
| {input_name}_hist_by_components.png | Component-wise density distribution histogram (generated when using --plot_all)                                                          |
| {input_name}_revised_contour.txt    | Text file containing revised contour levels (both conservative and aggressive) and masked percentage                                     |

When using --refinement_mask, additional files are generated:
| File                     | Description                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------|
| input_segment.mrc        | Input map after preprocessing for CryoREAD                                                   |
| mask_protein.mrc         | Initial protein mask from CryoREAD                                                           |
| chain_protein_prob.mrc   | Protein probability map from CryoREAD                                                        |
| chain_base_prob.mrc      | Base probability map from CryoREAD                                                           |
| chain_phosphate_prob.mrc | Phosphate probability map from CryoREAD                                                      |
| chain_sugar_prob.mrc     | Sugar probability map from CryoREAD                                                          |
| CCC_FSC05.txt            | Cross-Correlation between input map and masked volume and FSC 0.5 cutoff value from CryoREAD |

</details>

