# Introduction

A deep learning based tool to automatically select the best reconstructed 3D maps within a group of maps.

## Example Data

The repository includes example data in the `examples/` folder:
- `examples/job015/`: InitialModel job output with `initial_model.mrc` 
- `examples/job016/`: Class3D job output with 4 classes (`run_it025_class001.mrc` through `run_it025_class004.mrc`) and particle data (`run_it025_data.star`)

These examples can be used to test the functionality of DeepMASC without requiring your own data.

# Installation

<details>

### Clone the repository:

```bash
git clone https://github.com/AntiMatter568/DeepMASC
```

### Install conda/mamba:

If you don't have conda or mamba installed, choose one of the following options:

#### Option 1: Miniconda (Recommended)
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts and restart your terminal
```

#### Option 2: Miniforge (Community-driven, includes conda-forge by default + mamba)
```bash
# Download and install Miniforge (includes both conda and mamba)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# Follow the prompts and restart your terminal
```

For other operating systems, visit:
- [Miniconda Downloads](https://docs.conda.io/en/latest/miniconda.html)
- [Miniforge Releases](https://github.com/conda-forge/miniforge/releases)

### Create conda environment:

```bash
cd <Your Installation Folder> # default is DeepMASC
conda env create -f environment.yml

# If using miniforge with mamba (faster alternative):
mamba env create -f environment.yml
```

### Activate the environment:

```bash
conda activate DeepMASC

# Note: Environment activation is the same for both conda and mamba
```

### Configure config.py:

After setting up the conda environment, you need to configure the `config.py` file to point to your conda environment's Python executable:

1. **Find your conda environment's Python path:**
   ```bash
   # Method 1: Using conda info
   conda info --envs
   
   # Method 2: Activate environment and check Python path
   conda activate DeepMASC
   which python
   ```

2. **Edit config.py file:**
   Open `config.py` in the DeepMASC directory and update the `CONDA_PYTHON_PATH` variable:
   
   ```python
   # Update this line with your actual conda environment path
   CONDA_PYTHON_PATH = "/path/to/your/conda/envs/DeepMASC/bin/python"
   ```

### Install UCSF Chimera (Required):

UCSF Chimera is required for map resampling functionality in AutoContour and map visualization.

**Installation Options:**

1. **Download and install from official website:**
   - Visit: https://www.cgl.ucsf.edu/chimera/download.html
   - Download the appropriate version for your operating system
   - Follow the installation instructions for your platform

2. **For Linux users:**
   - Download the Linux version from the official website
   - Make the downloaded file executable and run the installer:
   ```bash
   # After downloading the .bin file from the website
   chmod +x chimera-*.bin
   ./chimera-*.bin
   ```

3. **Verify installation:**
   ```bash
   # Test that chimera command is available
   chimera --version
   
   # Test nogui mode (used by DeepMASC scripts)
   chimera --nogui --help
   ```

**Note:** 
- Chimera is **required** for AutoContour resampling functionality
- Make sure the `chimera` command is available in your system PATH
- DeepMASC will fail if Chimera is not properly installed when using AutoContour features

</details>

# AutoClass3D

## Standalone Usage

<details>

### Basic Usage

If you are in the repo directory:
```bash
conda activate DeepMASC
PYTHONNOUSERSITE=1 python main.py
```

If you want to run from arbitrary directory:
```bash
conda activate DeepMASC
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 python main.py
```

### Alternative: Using Direct Python Path (No Activation Required)

You can also run DeepMASC without activating the environment by using the direct path to the conda environment's Python interpreter:

```bash
# Find your conda/mamba environment path
conda info --envs
# OR if using mamba:
mamba info --envs

# Use the direct Python path (replace <CONDA_PATH> with your actual conda installation path)
<CONDA_PATH>/envs/DeepMASC/bin/python main.py
```

For example:
```bash
# Common conda paths:
# Miniconda: ~/miniconda3/envs/DeepMASC/bin/python main.py
# Miniforge: ~/miniforge3/envs/DeepMASC/bin/python main.py
# System conda: /opt/conda/envs/DeepMASC/bin/python main.py

~/miniconda3/envs/DeepMASC/bin/python main.py -f examples/job016/run_it025_class001.mrc -g 0 -o output_test
```

### Arguments

**Required Arguments:**
* `-f, --files`: List of input mrc files. Accepts multiple files separated by spaces. These are the MRC files that will be processed and selected.

* `-o, --output`: Output folder name. Directory where all output files will be stored.

* `-g, --gpus`: GPU ID to use for CryoREAD prediction. Specifies which GPU device should be used for processing. Multiple GPU IDs can be provided using a comma-separated list.

**Optional Arguments:**
* `--debug`: Enable debug mode to generate full output (default: False). When enabled, copy the full cryoREAD output to the output directory for debugging.

* `-r, --reso`: Resolution setting to choose the deep learning model (default: "Low", options are "Low" (>5Å) or "High" (<5Å)). Determines which model checkpoint will be used based on the desired resolution.

* `-b, --batch`: Batch size to use for CryoREAD prediction. Controls how many boxes are processed simultaneously during the prediction phase.

* `--dryrun`: When enabled, performs a dry run that only prints commands without actually executing CryoREAD. Useful for testing and verification.

### Examples

**Method 1: With environment activation**
```bash
conda activate DeepMASC
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 python main.py -f examples/job016/run_it025_class001.mrc examples/job016/run_it025_class002.mrc examples/job016/run_it025_class003.mrc examples/job016/run_it025_class004.mrc -g 0 -o output_autoselectclass
```

**Method 2: Direct Python path (no activation needed)**
```bash
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 ~/miniconda3/envs/DeepMASC/bin/python main.py -f examples/job016/run_it025_class001.mrc examples/job016/run_it025_class002.mrc examples/job016/run_it025_class003.mrc examples/job016/run_it025_class004.mrc -g 0 -o output_autoselectclass
```

**Using your own data:**
```bash
# Method 1: With activation
conda activate DeepMASC
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 python main.py -f /path/to/your/class1.mrc /path/to/your/class2.mrc /path/to/your/class3.mrc -g 0,1,2 -o /path/to/output/dir

# Method 2: Direct Python path
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 ~/miniconda3/envs/DeepMASC/bin/python main.py -f /path/to/your/class1.mrc /path/to/your/class2.mrc /path/to/your/class3.mrc -g 0,1,2 -o /path/to/output/dir
```

</details>

## RELION GUI Integration

<details>

### Files

There are three files associated with RELION integration of AutoSelect3D:

- `gtf_relion4_run_select_class3d.py` <- **this is the main file to execute**
- `gtf_relion4_create_select3d_data_star.py`
- `gtf_relion4_select3d.py`

### Arguments

**Required Arguments:**
* `-i, --input, --in_parts`: Input particle star file path (relative). This star file should come from RELION InitialModel/Class3D run and is automatically generated by RELION.

* `-o, --output`: Output job directory path (relative). This directory is automatically generated by RELION and will store all output files.

* `-g, --gpus`: GPU ID to use for CryoREAD prediction. Specifies which GPU device should be used for processing. Multiple GPU IDs can be provided using a comma-separated list.

**Optional Arguments:**
* `--debug`: Enable debug mode to generate full output (default: False). When enabled, copy the full cryoREAD output to the output directory for debugging.

* `-r, --reso`: Resolution setting to choose the deep learning model (default: "Low", options are "Low" (>5Å) or "High" (<5Å)). Determines which model checkpoint will be used based on the desired resolution.

* `-b, --batch`: Batch size to use for CryoREAD prediction. Controls how many boxes are processed simultaneously during the prediction phase.

### RELION GUI Setup Instructions

1. From RELION GUI, Choose "External", then in "External Executable" box enter:
   - **External Executable**: `python /path/to/gtf_relion4_run_select_class3d.py`

2. In the "Input" tab:
   - In "Input Particles" box, enter the path to the input data star file like `Class3D/job016/run_it025_data.star` (using the provided example) or your own `Class3D/jobXXX/run_it025_data.star`.

3. In the "Params" tab, you can set the following parameters:
   - `gpus`: GPU IDs to use for CryoREAD prediction (required), e.g., `0` or `0,1,2`
   - `debug`: Set to `True` to enable debug mode (optional)
   - `reso`: Resolution setting (optional, Low(>5Å) or High(<5Å))
   - `batch`: Batch size for CryoREAD prediction (optional, on modern GPU 8 and 16 works well)

4. In the "Running" tab:
   - Set "Number of threads" to 1
   - Adjust your submission to queue settings if using a managed queue system

5. Click the "Run" button to start the job.

6. Once finished, the results will be stored in the output job directory created by RELION.

</details>

## List of Output Files

<details>

| File                                                                                                                                    | Description                                                                                                                                      |
|-----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| {map_name}_CCC_FSC05.txt                                                                                                                | The first line is Real space CCC between CryoREAD prediction probabilities and input map, the second line is FSC 0.5 cutoff between the two maps |
| {map_name}_chain_base_prob.mrc, {map_name}_chain_phosphate_prob.mrc, {map_name}_chain_sugar_prob.mrc, {map_name}_chain_protein_prob.mrc | The predicted probabilities for nitrogenous bases, phosphate backbone, sugar ring, and protein                                                   |
| selected_model_map.mrc                                                                                                                  | The selected map based on maximum CCC criteria                                                                                                   |
| {map_name}_mask_protein.mrc                                                                                                             | The generated protein mask based on a custom cutoff value on protein probability                                                                 |
| {map_name}_segment.mrc                                                                                                                  | The resampled and cropped map used as input for CryoREAD                                                                                         |

</details>

# AutoContour

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
conda activate DeepMASC
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 python contour.py -i examples/job016/run_it025_class001.mrc -o output_autocontour_gmm -g 0 -p
```

Using your own data:
```bash
conda activate DeepMASC
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 python contour.py -i /path/to/your/map.mrc -o output_folder -g 0 -p
```

**CryoREAD Auto Refinement Masking:**

Using the provided example data:
```bash
conda activate DeepMASC
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 python contour.py -i examples/job016/run_it025_class001.mrc -o output_autocontour_cryoread -g 0 -p -r -b 16
```

Using your own data:
```bash
conda activate DeepMASC
cd /path/to/DeepMASC/repo
PYTHONNOUSERSITE=1 python contour.py -i /path/to/your/map.mrc -o output_folder -g 0 -p -r -b 16
```

</details>

## RELION GUI Integration

<details>

### Files

There is one file associated with RELION integration of AutoContour:

- `gtf_relion4_run_autocontour.py` <- **this is the main file to execute**

### RELION GUI Setup Instructions

1. From RELION GUI, Choose "External", then in "External Executable" box enter:
   - **External Executable**: `python /path/to/gtf_relion4_run_autocontour.py`

2. In the "Input" tab:
   - In "Reference map" box, select the map file you want to generate a mask for.

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


## Additional Notes for Conda Usage

<details>

### Environment Management

To list all conda environments:
```bash
conda env list
```

To remove the DeepMASC environment (if needed):
```bash
conda env remove -n DeepMASC
```

To update the environment (if environment.yml is updated):
```bash
conda env update -f environment.yml
```

### Running from Different Directories

To run DeepMASC from any directory:
1. Activate the conda environment: `conda activate DeepMASC`
2. Navigate to the DeepMASC repository directory: `cd /path/to/DeepMASC`
3. Run the scripts as shown in the examples above
4. Use relative or absolute paths for input/output files as needed

### Finding Your Conda Environment Path

To find the exact path to your conda environment's Python interpreter:

```bash
# Find all conda environments and their paths
conda info --envs

# Find the specific Python path for DeepMASC environment
conda info --envs | grep DeepMASC

# Alternative: activate environment and check Python path
conda activate DeepMASC
which python
```

Common conda installation paths:
- **Miniconda**: `~/miniconda3/envs/DeepMASC/bin/python`
- **Miniforge**: `~/miniforge3/envs/DeepMASC/bin/python`
- **System conda**: `/opt/conda/envs/DeepMASC/bin/python`
- **Anaconda (if installed)**: `~/anaconda3/envs/DeepMASC/bin/python`

</details>
