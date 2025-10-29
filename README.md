# DeepMASC

A deep learning based tool to automatically select the best reconstructed 3D maps within a group of maps.

## Features

- **AutoSelect3D**: Automatically select the best reconstructed 3D maps from RELION Class3D/InitialModel outputs
- **AutoContour**: Automatically determine contour levels and generate masks for cryo-EM maps
- **Mask Evaluation**: Evaluate 3D mask quality based on FSC criteria from RELION PostProcess jobs
- **Module System**: Optional environment module system for easy command-line access without manual activation

## Example Data

The repository includes example data in the `examples/` folder:
- `examples/InitialModel/job015/`: InitialModel job output with `initial_model.mrc` 
- `examples/Class3D/job016/`: Class3D job output with 4 classes (`run_it025_class001.mrc` through `run_it025_class004.mrc`) and particle data (`run_it025_data.star`)

These examples can be used to test the functionality of DeepMASC without requiring your own data.

**Credit**: Example data is taken from the [RELION Single Particle Analysis Tutorial](https://relion.readthedocs.io/en/latest/SPA_tutorial/index.html).

## Installation

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

### Activate the environment (optional):

```bash
# Optional: You can activate the environment if you prefer
# conda activate DeepMASC

# Note: Environment activation is the same for both conda and mamba
# Alternatively, you can use the direct Python path as shown in examples
```

### Configure config.py:

After setting up the conda environment, you need to configure the `config.py` file to point to your conda environment's Python executable:

1. **Find your conda environment's Python path:**
   ```bash
   # Method 1: Using conda info
   conda info --envs
   
   # Method 2: Check Python path after activating environment
   # conda activate DeepMASC
   # which python
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

## Documentation

- **[AutoSelect3D Documentation](README_AutoSelect3D.md)** - Automatically select the best 3D maps from Class3D/InitialModel outputs
- **[AutoContour Documentation](README_AutoContour.md)** - Automatically determine contour levels and generate masks
- **[Mask Evaluation Documentation](README_MaskEvaluation.md)** - Evaluate 3D mask quality based on FSC criteria
- **[Module System Guide](module/README.md)** - Quick start guide for using DeepMASC with environment modules
- **[Module Installation Guide](module/INSTALL.md)** - Detailed installation instructions for the module system