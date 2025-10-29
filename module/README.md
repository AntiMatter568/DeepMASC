# DeepMASC Module System

This directory contains a module system for DeepMASC that allows you to run the tools without activating the conda environment or modifying your system PATH permanently.

## Quick Start

The easiest way to set this up is to run the setup script:

```bash
cd <DeepMASC>/module
./setup.sh
```

Replace `<DeepMASC>` with your actual DeepMASC installation path.

The setup script will:
1. Auto-detect your conda installation
2. Auto-detect your module system directory
3. Configure the wrapper scripts and module files
4. Guide you through personal or system-wide installation

## What's Included

### Files

- **`deepmasc.tcl`** - Tcl modulefile for Environment Modules
- **`setup.sh`** - Interactive setup script (recommended)
- **`uninstall.sh`** - Automatic uninstall script
- **`INSTALL.md`** - Detailed installation instructions
- **`bin/deepmasc`** - Wrapper script for main.py (AutoSelect3D)
- **`bin/deepmasc-contour`** - Wrapper script for contour.py (AutoContour)
- **`bin/deepmasc-relion-select`** - Wrapper for RELION AutoSelect3D
- **`bin/deepmasc-relion-contour`** - Wrapper for RELION AutoContour
- **`bin/deepmasc-relion-eval-mask`** - Wrapper for RELION Eval Refinement Mask

### Commands

After installation, you'll have access to:

**Standalone Commands:**
- `deepmasc` - Run AutoSelect3D to select the best 3D map
- `deepmasc-contour` - Run AutoContour to generate masks

**RELION Integration Commands:**
- `deepmasc-relion-select` - AutoSelect3D for RELION GUI (replaces `gtf_relion4_run_select_class3d.py`)
- `deepmasc-relion-contour` - AutoContour for RELION GUI (replaces `gtf_relion4_run_autocontour.py`)
- `deepmasc-relion-eval-mask` - Eval Refinement Mask for RELION GUI (replaces `gtf_relion4_run_eval_refinement_mask.py`)

## Installation

### Automatic Setup (Recommended)
```bash
./setup.sh
```

The setup script will auto-detect your system configuration and guide you through the installation process.

### Manual Installation
See [INSTALL.md](INSTALL.md) for detailed manual installation instructions if you prefer to configure everything yourself.

## Usage Examples

### Standalone Usage

```bash
# Load the module
module load deepmasc

# AutoSelect3D - Select best map from multiple classes
deepmasc -f examples/job016/run_it025_class*.mrc -g 0 -o output

# AutoContour - Generate masks
deepmasc-contour -i examples/job016/run_it025_class001.mrc -o mask_output -g 0

# Unload when done
module unload deepmasc
```

### RELION GUI Integration

After loading the module, you can use the wrapper commands in RELION's External job:

#### AutoSelect3D in RELION
```
External Executable: deepmasc-relion-select
```

Then in RELION GUI:
- **Input tab**: Set "Input particles" to your `Class3D/jobXXX/run_it025_data.star`
- **Params tab**: Add `gpus` parameter (e.g., `0` or `0,1`)
- **Running tab**: Set "Number of threads" to `1`

#### AutoContour in RELION
```
External Executable: deepmasc-relion-contour
```

Then in RELION GUI:
- **Input tab**: Set "Reference map" to your map file
- **Params tab**: Add `gpus` parameter (e.g., `0`)
- **Running tab**: Set "Number of threads" to `1`

#### Eval Refinement Mask in RELION
```
External Executable: deepmasc-relion-eval-mask
```

See the main [README.md](../README.md) for detailed RELION integration instructions.

## Requirements

- DeepMASC conda environment must be created (`conda env create -f environment.yml`)
- Environment Modules software must be installed (see [INSTALL.md](INSTALL.md) for installation instructions)

## Troubleshooting

If you encounter issues, see the [INSTALL.md](INSTALL.md) file for troubleshooting tips, or:

1. Verify conda environment exists: `conda env list | grep DeepMASC`
2. Check Python path is correct in the wrapper scripts
3. Ensure wrapper scripts are executable: `ls -l bin/`
4. For modules: Check MODULEPATH includes this directory: `echo $MODULEPATH`

## Customization

You can add more wrapper scripts for other Python scripts in the DeepMASC project:

1. Create a new script in `bin/` directory
2. Copy the structure from existing wrapper scripts
3. Update the Python script path
4. Make it executable: `chmod +x bin/your-new-command`

## Uninstallation

### Automatic Uninstall (Recommended)

Run the automatic uninstall script:
```bash
cd <DeepMASC>/module
./uninstall.sh
```

The script will:
- Auto-detect your installation type (personal or system-wide)
- Remove module files and shell configuration entries
- Provide clear instructions for any manual steps needed

### Manual Uninstall

For detailed manual removal instructions, see [INSTALL.md](INSTALL.md#uninstallation).

## Support

For more information about DeepMASC itself, see the main [README.md](../README.md) in the project root.
