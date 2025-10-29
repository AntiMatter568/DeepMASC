#%Module1.0
##
## DeepMASC modulefile
##
## This module provides access to the DeepMASC tools without adding
## the conda environment to your PATH
##

# Get the directory where this module file is located
# If it's a symlink, resolve it to the actual file location
set modulefile $ModulesCurrentModulefile
if { [file type $modulefile] eq "link" } {
    set modulefile [file readlink $modulefile]
}
set moduledir [file dirname $modulefile]
set basedir [file dirname $moduledir]

# Set conda environment name and base path
# IMPORTANT: Update CONDA_PREFIX to point to your conda installation
set conda_base "/home/zpahai/miniforge3"
set env_name "DeepMASC"
set conda_env "${conda_base}/envs/${env_name}"

# Check if conda environment exists, if not try common locations
if { ![file exists "$conda_env"] } {
    if { [file exists "$env(HOME)/anaconda3/envs/${env_name}"] } {
        set conda_env "$env(HOME)/anaconda3/envs/${env_name}"
    } elseif { [file exists "$env(HOME)/mambaforge/envs/${env_name}"] } {
        set conda_env "$env(HOME)/mambaforge/envs/${env_name}"
    } elseif { [file exists "/opt/conda/envs/${env_name}"] } {
        set conda_env "/opt/conda/envs/${env_name}"
    }
}

proc ModulesHelp { } {
    global basedir
    puts stderr "\tDeepMASC - Deep Learning for Map Selection and Contouring"
    puts stderr "\n\tStandalone Commands:"
    puts stderr "\t  deepmasc                    - Main AutoSelect3D tool"
    puts stderr "\t  deepmasc-contour            - AutoContour mask generation tool"
    puts stderr "\n\tRELION Integration Commands:"
    puts stderr "\t  deepmasc-relion-select      - AutoSelect3D for RELION GUI"
    puts stderr "\t  deepmasc-relion-contour     - AutoContour for RELION GUI"
    puts stderr "\t  deepmasc-relion-eval-mask   - Eval Refinement Mask for RELION GUI"
    puts stderr "\n\tProject directory: $basedir"
}

module-whatis "DeepMASC - Deep Learning for cryo-EM map selection and contouring"

# Add wrapper scripts to PATH
prepend-path PATH ${basedir}/module/bin

# Set environment variables
setenv DEEPMASC_ROOT ${basedir}
setenv DEEPMASC_CONDA_ENV ${conda_env}
setenv DEEPMASC_PYTHON ${conda_env}/bin/python
setenv PYTHONNOUSERSITE 1

# Display load message
if { [module-info mode load] } {
    puts stderr "DeepMASC loaded"
    puts stderr "Standalone: deepmasc, deepmasc-contour"
    puts stderr "RELION: deepmasc-relion-select, deepmasc-relion-contour, deepmasc-relion-eval-mask"
    puts stderr "Using conda environment: ${conda_env}"
}

# Display unload message
if { [module-info mode remove] } {
    puts stderr "DeepMASC unloaded"
}
