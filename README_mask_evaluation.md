# RELION External Script: Mask3D Evaluation

This script evaluates the quality of 3D masks used in RELION refinement based on FSC (Fourier Shell Correlation) criteria.

## Overview

The script analyzes FSC curves from RELION PostProcess jobs to determine if a mask meets the quality criteria defined in Method C of mask evaluation. It checks whether the first zero crossing of the phase randomized FSC curve occurs at lower resolution than the FSC=0.5 criterion without any mask.

## Usage with RELION GUI

### Setup Instructions

1. **From RELION GUI**, choose "External" job type
2. **In "External Executable" box**, enter:
   ```
   python /path/to/gtf_relion4_run_eval_refinement_mask.py
   ```

3. **In the "Input" tab**:
   - **Input PostProcess**: Select the PostProcess star file (e.g., `PostProcess/job123/postprocess.star`)

4. **In the "Params" tab**, you can set the following optional parameters:
   - `plot`: Set to `True` to generate FSC curves plot (default: True)
   - `debug`: Set to `True` to enable debug mode (default: False)

5. **In the "Running" tab**:
   - Set "Number of threads" to 1
   - Adjust submission settings if using a queue system

6. **Click "Run"** to start the evaluation

### Command Line Usage

For direct command line usage:

```bash
python gtf_relion4_run_eval_refinement_mask.py \
    -i PostProcess/job123/postprocess.star \
    -o External/job456/ \
    --plot True \
    --debug False
```

## Input Requirements

- **PostProcess Star File**: A RELION PostProcess job output containing FSC data with the following columns:
  - `rlnAngstromResolution`
  - `rlnFourierShellCorrelationUnmaskedMaps`
  - `rlnFourierShellCorrelationMaskedMaps`
  - `rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps`
  - `rlnFourierShellCorrelationCorrected`

## Output Files

The script generates the following output files in the RELION job directory:

| File | Description |
|------|-------------|
| `mask3d_evaluation.csv` | CSV file containing evaluation results and metrics |
| `mask3d_evaluation_*.png` | FSC curves plot (if plotting enabled) |
| `mask_evaluation_summary.star` | RELION-compatible summary star file |
| `RELION_JOB_EXIT_SUCCESS` | RELION success indicator file |
| `RELION_OUTPUT_NODES.star` | RELION pipeline nodes file |

## Evaluation Criteria

The script evaluates masks based on the following criteria:

1. **Resolution at FSC=0.5 (unmasked)**: The resolution where the unmasked FSC drops below 0.5
2. **Phase randomized FSC zero crossing**: The first resolution where phase randomized FSC crosses zero
3. **Resolution at FSC=0.143 (corrected)**: The resolution where corrected FSC drops below 0.143

**PASS Criterion**: The phase randomized FSC zero crossing occurs at lower resolution (higher Å value) than the FSC=0.5 criterion without mask.

## Interpretation of Results

### CSV Output

The `mask3d_evaluation.csv` file contains:
- `unmasked_res_0_5`: Resolution at FSC=0.5 without mask (Å)
- `phase_rand_zero_res`: Resolution at phase randomized FSC zero crossing (Å)
- `corrected_res_0_143`: Resolution at FSC=0.143 with corrected FSC (Å)
- `criterion_met`: Boolean indicating if the mask passes the evaluation
- `valid`: Boolean indicating if all measurements were valid

### Plot Output

The FSC curves plot shows:
- **Blue line**: Unmasked FSC
- **Orange line**: Masked FSC
- **Green line**: Phase Randomized FSC
- **Red line**: Corrected FSC
- **Vertical lines**: Key resolution thresholds
- **Annotations**: Resolution values at important thresholds

### Summary Star File

The `mask_evaluation_summary.star` file contains RELION-compatible results that can be used in downstream processing or analysis.