#!/usr/bin/env bash
#
# DeepMASC Module Uninstall Script
# This script helps remove the DeepMASC module system installation
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "DeepMASC Module Uninstall"
echo "========================================="
echo ""

# Check if module is loaded
if command -v module &> /dev/null; then
    if module list 2>&1 | grep -q "deepmasc"; then
        echo "Unloading deepmasc module..."
        module unload deepmasc || true
    fi
fi

# Detect installation type
echo "Detecting installation type..."
echo ""

FOUND_PERSONAL=false
FOUND_SYSTEM=false
SYSTEM_MODULE_PATH=""

# Check for personal installation (in shell config)
if grep -q "MODULEPATH.*DeepMASC/module" ~/.bashrc 2>/dev/null; then
    FOUND_PERSONAL=true
fi

# Check for system-wide installation
COMMON_PATHS=(
    "/usr/share/modules/modulefiles/deepmasc"
    "/usr/share/modulefiles/deepmasc"
    "/etc/modulefiles/deepmasc"
    "/usr/local/modules/modulefiles/deepmasc"
    "/opt/modules/modulefiles/deepmasc"
)

for path in "${COMMON_PATHS[@]}"; do
    if [ -d "$path" ] || [ -L "$path" ]; then
        FOUND_SYSTEM=true
        SYSTEM_MODULE_PATH="$path"
        break
    fi
done

# Display findings
if [ "$FOUND_PERSONAL" = false ] && [ "$FOUND_SYSTEM" = false ]; then
    echo "No DeepMASC module installation detected."
    echo "The module system may have already been uninstalled."
    echo ""
    read -p "Do you want to clean up the module directory anyway? [y/N]: " cleanup
    if [[ $cleanup =~ ^[Yy]$ ]]; then
        echo ""
        echo "Module directory location: $SCRIPT_DIR"
        echo "You can manually remove it with: rm -rf \"$SCRIPT_DIR\""
    fi
    exit 0
fi

echo "Found the following installations:"
if [ "$FOUND_PERSONAL" = true ]; then
    echo "  ✓ Personal installation (in ~/.bashrc)"
fi
if [ "$FOUND_SYSTEM" = true ]; then
    echo "  ✓ System-wide installation at: $SYSTEM_MODULE_PATH"
fi
echo ""

# Confirm uninstallation
read -p "Do you want to proceed with uninstallation? [y/N]: " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

echo ""

# Remove personal installation
if [ "$FOUND_PERSONAL" = true ]; then
    echo "Removing personal installation..."

    # Backup bashrc
    cp ~/.bashrc ~/.bashrc.backup.deepmasc
    echo "✓ Created backup: ~/.bashrc.backup.deepmasc"

    # Remove lines related to DeepMASC module
    sed -i '/MODULEPATH.*DeepMASC\/module/d' ~/.bashrc
    echo "✓ Removed DeepMASC module entries from ~/.bashrc"
    echo ""
    echo "To restore your original .bashrc, run:"
    echo "  mv ~/.bashrc.backup.deepmasc ~/.bashrc"
fi

# Remove system-wide installation
if [ "$FOUND_SYSTEM" = true ]; then
    echo "Removing system-wide installation..."

    read -p "Remove $SYSTEM_MODULE_PATH? This requires sudo. [y/N]: " confirm_sudo
    if [[ $confirm_sudo =~ ^[Yy]$ ]]; then
        sudo rm -rf "$SYSTEM_MODULE_PATH"
        echo "✓ Removed $SYSTEM_MODULE_PATH"
    else
        echo "Skipped system-wide removal. To remove manually, run:"
        echo "  sudo rm -rf \"$SYSTEM_MODULE_PATH\""
    fi
fi

echo ""
echo "========================================="
echo "Uninstallation Complete"
echo "========================================="
echo ""

# Verify
echo "Verifying removal..."
if command -v module &> /dev/null; then
    if module avail deepmasc 2>&1 | grep -q deepmasc; then
        echo "⚠ Warning: 'module avail deepmasc' still shows the module"
        echo "  You may need to restart your terminal or run: source ~/.bashrc"
    else
        echo "✓ Module no longer appears in 'module avail'"
    fi
fi

if command -v deepmasc &> /dev/null; then
    echo "⚠ Warning: 'deepmasc' command still found in PATH"
    echo "  You may need to restart your terminal or run: source ~/.bashrc"
else
    echo "✓ Commands removed from PATH"
fi

echo ""
echo "Notes:"
echo "  - The DeepMASC conda environment was NOT removed"
echo "  - The main DeepMASC code was NOT removed"
echo "  - Only the module system wrapper was uninstalled"
echo ""
echo "To remove the DeepMASC conda environment:"
echo "  conda env remove -n DeepMASC"
echo ""
echo "Please restart your terminal or run 'source ~/.bashrc' to complete the removal."
