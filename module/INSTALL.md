# DeepMASC Module Installation Guide

This guide explains how to install and use the DeepMASC environment module system.

**Recommended:** Use the automatic setup script (`./setup.sh`) which auto-detects your system configuration. Manual instructions are provided below for advanced users.

## What This Provides

The module system provides command-line tools to run DeepMASC without:
- Activating the conda environment manually
- Adding the Python interpreter to your system PATH
- Typing long paths to Python scripts

## Available Commands

After loading the module, you'll have access to:

**Standalone Commands:**
- `deepmasc` - Runs the main AutoSelect3D tool (main.py)
- `deepmasc-contour` - Runs the AutoContour mask generation tool (contour.py)

**RELION Integration Commands:**
- `deepmasc-relion-select` - AutoSelect3D for RELION GUI
- `deepmasc-relion-contour` - AutoContour for RELION GUI
- `deepmasc-relion-eval-mask` - Eval Refinement Mask for RELION GUI

These RELION commands can be used directly in RELION's "External" job type without needing full Python paths.

## Quick Installation (Recommended)

Run the automatic setup script:
```bash
cd <DeepMASC>/module
./setup.sh
```

The setup script will:
- Auto-detect your conda installation
- Auto-detect your module system directory
- Configure everything automatically
- Prompt you for installation preferences

## Manual Installation

### Using Environment Modules

#### Step 1: Install Environment Modules (if not already installed)

On Ubuntu/Debian:
```bash
sudo apt-get install environment-modules
```

On Red Hat/CentOS/Fedora:
```bash
sudo yum install environment-modules
# or
sudo dnf install environment-modules
```

#### Step 2: Update the modulefile with your conda path

Edit the modulefile to point to your conda installation:
```bash
nano <DeepMASC>/module/deepmasc.tcl
```

Replace `<DeepMASC>` with your actual DeepMASC installation path.

Find this line:
```tcl
set conda_base "$env(HOME)/miniconda3"
```

Change it to your actual conda installation path, for example:
```tcl
set conda_base "$env(HOME)/anaconda3"
# or
set conda_base "$env(HOME)/mambaforge"
# or
set conda_base "/opt/conda"
```

#### Step 3: Add the module to your module path

##### For personal use:
```bash
# Add to your ~/.bashrc or ~/.bash_profile
# Replace <DeepMASC> with your actual DeepMASC installation path
echo 'export MODULEPATH=$MODULEPATH:<DeepMASC>/module' >> ~/.bashrc
source ~/.bashrc
```

##### For system-wide installation (requires root):
```bash
# First, find your system's modulefile directory:
# - Check: module avail 2>&1 | grep modulefiles
# - Or check: echo $MODULEPATH
# - Common locations: /usr/share/modules/modulefiles, /usr/share/modulefiles, /etc/modulefiles

# Once you know the path, create the directory and symlink:
# Replace <DeepMASC> with your actual DeepMASC installation path
# Replace <MODULEFILE_DIR> with your system's modulefile directory
sudo mkdir -p <MODULEFILE_DIR>/deepmasc
sudo ln -s <DeepMASC>/module/deepmasc.tcl <MODULEFILE_DIR>/deepmasc/1.0

# Or copy it
sudo cp <DeepMASC>/module/deepmasc.tcl <MODULEFILE_DIR>/deepmasc/1.0
```

#### Step 4: Load and use the module

```bash
# Load the module
module load deepmasc

# Check available commands
deepmasc --help
deepmasc-contour --help

# Run DeepMASC standalone
deepmasc -f examples/Class3D/job016/run_it025_class*.mrc -g 0 -o output_test

# Or use in RELION GUI - just specify the command name:
# External Executable: deepmasc-relion-select
# External Executable: deepmasc-relion-contour
# External Executable: deepmasc-relion-eval-mask

# Unload when done (optional)
module unload deepmasc
```

## Usage Examples

### AutoSelect3D - Select best 3D map from multiple classes

```bash
# After loading the module
deepmasc \
  -f examples/Class3D/job016/run_it025_class001.mrc \
     examples/Class3D/job016/run_it025_class002.mrc \
     examples/Class3D/job016/run_it025_class003.mrc \
     examples/Class3D/job016/run_it025_class004.mrc \
  -g 0 \
  -o output_autoselect
```

### AutoContour - Generate masks

```bash
# GMM-based mask (fast)
deepmasc-contour \
  -i examples/Class3D/job016/run_it025_class001.mrc \
  -o output_mask_gmm \
  -g 0 -p

# CryoREAD refinement mask (slower, more accurate)
deepmasc-contour \
  -i examples/Class3D/job016/run_it025_class001.mrc \
  -o output_mask_refined \
  -g 0 -p -r -b 16
```

## Module Commands (Option 1 only)

```bash
# List available modules
module avail

# Show information about the DeepMASC module
module show deepmasc
module help deepmasc

# Load the module
module load deepmasc

# Check what modules are loaded
module list

# Unload the module
module unload deepmasc
```

## Verification

To verify the installation works:

```bash
# Load the module
module load deepmasc

# Verify commands are available
which deepmasc
which deepmasc-contour

# Test the commands
deepmasc --help
deepmasc-contour --help
```

## Troubleshooting

### "module: command not found"
- Environment modules is not installed. Install it using your package manager (see installation instructions above).

### "DEEPMASC_PYTHON not set"
- You forgot to load the module. Run: `module load deepmasc`

### "Python interpreter not found"
- The conda environment path is incorrect in the modulefile.
- Find your conda environment path: `conda info --envs | grep DeepMASC`
- Update the path in the modulefile: `nano <DeepMASC>/module/deepmasc.tcl`

### "No module named 'torch'" or similar import errors
- The conda environment is not properly set up.
- Make sure you created the environment: `conda env create -f environment.yml`
- Verify the environment exists: `conda env list`

## Uninstallation

### Automatic Uninstall (Recommended)

The easiest way to uninstall is to use the automatic uninstall script:

```bash
cd <DeepMASC>/module
./uninstall.sh
```

The script will:
- Auto-detect your installation type (personal or system-wide)
- Unload the module if loaded
- Remove configuration entries from ~/.bashrc (with backup)
- Remove system-wide module files (with confirmation)
- Verify the removal was successful

### Manual Uninstall

If you prefer to uninstall manually, follow the instructions below:

#### Removing Environment Modules Installation

##### For personal installation:
```bash
# 1. Unload the module if currently loaded
module unload deepmasc

# 2. Remove the MODULEPATH entry from your shell config
# Edit ~/.bashrc or ~/.bash_profile and remove the line:
#   export MODULEPATH=$MODULEPATH:<DeepMASC>/module
nano ~/.bashrc

# 3. Reload your shell configuration
source ~/.bashrc
```

##### For system-wide installation:
```bash
# 1. Unload the module if currently loaded
module unload deepmasc

# 2. Find where the module is installed
module show deepmasc 2>&1 | grep -i "deepmasc.tcl"
# Or check common locations:
ls /usr/share/modules/modulefiles/deepmasc/
ls /usr/share/modulefiles/deepmasc/

# 3. Remove the module directory (requires sudo)
# Replace <MODULEFILE_DIR> with the actual path found above
sudo rm -rf <MODULEFILE_DIR>/deepmasc

# Example:
sudo rm -rf /usr/share/modules/modulefiles/deepmasc

# 4. Verify removal
module avail deepmasc  # Should show no results
```

### Complete Removal (Optional)

If you want to completely remove the module system files from DeepMASC:

```bash
# Remove the module directory (this does not affect the main DeepMASC code)
rm -rf <DeepMASC>/module
```

**Note:** This only removes the module system wrapper. The main DeepMASC code and conda environment remain intact.

## Additional Notes

- The wrapper scripts automatically set `PYTHONNOUSERSITE=1` to avoid conflicts with user-installed Python packages.
- You can add more wrapper scripts to `<DeepMASC>/module/bin/` for other Python scripts in the project.
- The modulefile can be customized to set additional environment variables or aliases as needed.
