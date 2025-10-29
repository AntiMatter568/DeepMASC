#!/usr/bin/env bash
#
# DeepMASC Module Setup Script
# This script helps configure the module system for your environment
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPMASC_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "DeepMASC Module Setup"
echo "========================================="
echo ""

# Function to detect conda installation
detect_conda() {
    local conda_paths=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/mambaforge"
        "$HOME/miniforge3"
        "/opt/conda"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
    )

    for path in "${conda_paths[@]}"; do
        if [ -d "$path/envs/DeepMASC" ]; then
            echo "$path"
            return 0
        fi
    done

    # Try using conda command if available
    if command -v conda &> /dev/null; then
        local env_path=$(conda env list | grep "^DeepMASC " | awk '{print $NF}')
        if [ -n "$env_path" ]; then
            # Extract base conda path (remove /envs/DeepMASC)
            echo "${env_path%/envs/DeepMASC}"
            return 0
        fi
    fi

    return 1
}

# Detect conda installation
echo "Step 1: Detecting conda installation..."
if CONDA_BASE=$(detect_conda); then
    echo "✓ Found conda installation: $CONDA_BASE"
    echo "✓ DeepMASC environment found: $CONDA_BASE/envs/DeepMASC"
    CONDA_ENV="$CONDA_BASE/envs/DeepMASC"
else
    echo "✗ Could not automatically detect DeepMASC conda environment"
    echo ""
    echo "Please enter the full path to your conda base directory:"
    echo "Examples:"
    echo "  $HOME/miniconda3"
    echo "  $HOME/anaconda3"
    echo "  /opt/conda"
    echo ""
    read -p "Conda base path: " CONDA_BASE
    CONDA_ENV="$CONDA_BASE/envs/DeepMASC"

    if [ ! -d "$CONDA_ENV" ]; then
        echo "✗ Error: DeepMASC environment not found at: $CONDA_ENV"
        echo ""
        echo "Please create the environment first:"
        echo "  conda env create -f $DEEPMASC_ROOT/environment.yml"
        exit 1
    fi
fi

echo ""
echo "Step 2: Setting up Environment Modules..."
echo ""

# Check if environment modules is installed
if ! command -v modulecmd &> /dev/null; then
    echo "⚠ Warning: Environment Modules doesn't appear to be installed"
    echo ""
    echo "To install Environment Modules:"
    echo "  Ubuntu/Debian: sudo apt-get install environment-modules"
    echo "  RHEL/CentOS:   sudo yum install environment-modules"
    echo ""
    read -p "Continue anyway? [y/N]: " continue
    if [[ ! $continue =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect module system's modulefile directory
echo "Detecting module system location..."
SYSTEM_MODULEFILE_DIR=""

# Method 1: Parse module avail output to find existing modulefiles
if command -v module &> /dev/null; then
    # Capture the header line from module avail that shows the path
    MODULE_AVAIL_OUTPUT=$(module avail 2>&1)

    # Look for lines that contain directory paths (typically shown as headers)
    # Example: "------- /usr/share/modules/modulefiles -------"
    DETECTED_PATH=$(echo "$MODULE_AVAIL_OUTPUT" | grep -oE '(/[^ ]*modulefiles)' | head -1)

    if [ -n "$DETECTED_PATH" ] && [ -d "$DETECTED_PATH" ]; then
        SYSTEM_MODULEFILE_DIR="$DETECTED_PATH"
    fi
fi

# Method 2: Check MODULEPATH environment variable
if [ -z "$SYSTEM_MODULEFILE_DIR" ] && [ -n "$MODULEPATH" ]; then
    # Get the first path from MODULEPATH that exists and looks like a system directory
    IFS=':' read -ra PATHS <<< "$MODULEPATH"
    for path in "${PATHS[@]}"; do
        # Prefer system paths over user paths
        if [[ "$path" =~ ^/(usr|opt|etc) ]] && [ -d "$path" ]; then
            SYSTEM_MODULEFILE_DIR="$path"
            break
        fi
    done
    # If no system path found, use the first valid path
    if [ -z "$SYSTEM_MODULEFILE_DIR" ]; then
        for path in "${PATHS[@]}"; do
            if [ -d "$path" ]; then
                SYSTEM_MODULEFILE_DIR="$path"
                break
            fi
        done
    fi
fi

# Method 3: Check MODULESHOME
if [ -z "$SYSTEM_MODULEFILE_DIR" ] && [ -n "$MODULESHOME" ]; then
    if [ -d "$MODULESHOME/modulefiles" ]; then
        SYSTEM_MODULEFILE_DIR="$MODULESHOME/modulefiles"
    fi
fi

# Method 4: Try common system locations
if [ -z "$SYSTEM_MODULEFILE_DIR" ]; then
    COMMON_PATHS=(
        "/usr/share/modules/modulefiles"
        "/usr/share/modulefiles"
        "/etc/modulefiles"
        "/usr/local/modules/modulefiles"
        "/opt/modules/modulefiles"
        "/opt/modulefiles"
    )

    for path in "${COMMON_PATHS[@]}"; do
        if [ -d "$path" ]; then
            SYSTEM_MODULEFILE_DIR="$path"
            break
        fi
    done
fi

if [ -n "$SYSTEM_MODULEFILE_DIR" ]; then
    echo "✓ Detected module system directory: $SYSTEM_MODULEFILE_DIR"
else
    echo "⚠ Could not auto-detect module system directory"
fi

# Update modulefile with detected conda path
echo "Updating modulefile with conda path..."
sed -i "s|set conda_base \".*\"|set conda_base \"$CONDA_BASE\"|" "$SCRIPT_DIR/deepmasc.tcl"
echo "✓ Modulefile updated: $SCRIPT_DIR/deepmasc.tcl"

echo ""
echo "Step 3: Adding module to MODULEPATH..."
echo "Choose module path scope:"
echo "  1) Personal (~/.bashrc) - Only for your user"
echo "  2) System-wide (requires sudo) - For all users"
echo "  3) Skip - I'll do this manually"
echo ""
read -p "Enter choice [1, 2, or 3]: " path_choice

case $path_choice in
    1)
        echo "export MODULEPATH=\$MODULEPATH:$SCRIPT_DIR" >> ~/.bashrc
        echo "✓ Added to ~/.bashrc"
        echo ""
        echo "Run 'source ~/.bashrc' or restart your terminal to use the module"
        ;;
    2)
        if [ -z "$SYSTEM_MODULEFILE_DIR" ]; then
            echo "✗ Error: Could not detect module system location"
            echo ""
            echo "Please use option 1 (Personal installation) or option 3 (Manual),"
            echo "or manually specify the modulefile directory."
            echo ""
            read -p "Enter modulefile directory path (or press Enter to skip): " SYSTEM_MODULEFILE_DIR
            if [ -z "$SYSTEM_MODULEFILE_DIR" ]; then
                echo "Skipping system-wide installation."
                exit 1
            fi
        fi

        echo "Creating system module directory..."
        sudo mkdir -p "$SYSTEM_MODULEFILE_DIR/deepmasc"
        sudo ln -sf "$SCRIPT_DIR/deepmasc.tcl" "$SYSTEM_MODULEFILE_DIR/deepmasc/1.0"
        echo "✓ Module installed system-wide at: $SYSTEM_MODULEFILE_DIR/deepmasc/1.0"
        echo ""
        echo "All users can now run: module load deepmasc"
        ;;
    3)
        echo ""
        echo "To manually add the module, add this to your shell config:"
        echo "  export MODULEPATH=\$MODULEPATH:$SCRIPT_DIR"
        ;;
    *)
        echo "Invalid choice, skipping..."
        ;;
esac

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "To use DeepMASC:"
echo "  1. Load the module: module load deepmasc"
echo "  2. Run commands: deepmasc --help"
echo ""

echo "For more information, see: $SCRIPT_DIR/INSTALL.md"
