import os
import sys


def resample_single_map(input_map, reference_map, output_map):
    """
    Resample a single MRC map using ChimeraX to match a reference grid.

    Parameters:
    -----------
    input_map : str
        Path to input MRC file to resample
    reference_map : str
        Path to reference MRC file with target grid
    output_map : str
        Path to save resampled output file
    """
    from chimera import runCommand as rc

    # Check if input files exist
    if not os.path.exists(input_map):
        raise FileNotFoundError("Input map not found: {}".format(input_map))

    if not os.path.exists(reference_map):
        raise FileNotFoundError("Reference map not found: {}".format(reference_map))

    # Create output directory if needed
    output_dir = os.path.dirname(output_map)
    if output_dir and not os.path.exists(output_dir):
        print("Creating output directory: {}".format(output_dir))
        try:
            os.makedirs(output_dir)
            print("Directory created successfully")
        except OSError as e:
            print("Error creating directory: {}".format(e))
            raise

    print("Resampling: {}".format(input_map))
    print("Reference: {}".format(reference_map))
    print("Output: {}".format(output_map))
    print("Output directory: {}".format(os.path.dirname(output_map)))

    try:
        # Open input and reference maps
        rc("open {}".format(input_map))  # #0
        rc("open {}".format(reference_map))  # #1

        # Set step size to 1 for both volumes
        rc("volume #0 step 1")
        rc("volume #1 step 1")

        # Resample input map onto reference grid
        rc("vop resample #0 onGrid #1")  # Creates #2
        rc("volume #2 step 1")

        # Save resampled map
        print("About to save to: {}".format(output_map))
        print("Directory exists: {}".format(os.path.exists(os.path.dirname(output_map))))
        rc("volume #2 save {}".format(output_map))

        # Clean up
        rc("close all")

        print("Resampling completed successfully")

    except Exception as e:
        rc("close all")  # Clean up on error
        raise Exception("Error during resampling: {}".format(e))


# Command line argument parsing and execution
print("Command line arguments: {}".format(sys.argv))

# Handle Chimera's command line structure
# Expected: [__main__.py, --nogui, script_name, input_map, reference_map, output_map]
if len(sys.argv) >= 6:
    input_map = sys.argv[3]
    reference_map = sys.argv[4] 
    output_map = sys.argv[5]
else:
    sys.stderr.write("Error: Expected 3 arguments (input_map, reference_map, output_map)\n")
    sys.stderr.write("Usage: chimera --nogui resample_chimera.py input_map reference_map output_map\n")
    sys.exit(1)

print("Parsed arguments:")
print("  input_map: {}".format(input_map))
print("  reference_map: {}".format(reference_map))
print("  output_map: {}".format(output_map))

try:
    resample_single_map(os.path.abspath(input_map), os.path.abspath(reference_map), os.path.abspath(output_map))
except Exception as e:
    sys.stderr.write("Error: {}\n".format(e))
    sys.exit(1)
