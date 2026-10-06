# RELION External Script: Mask3D Evaluation

This script evaluates the quality of 3D masks used in RELION refinement based on FSC (Fourier Shell Correlation) criteria.

## Overview

The script analyzes FSC curves from RELION PostProcess jobs and reports whether the mask passes the PRFSC criteria. A mask passes when both of the following hold:

1. The phase randomized masked FSC reaches 0 at or before the shell of the masked 0.143 resolution. The shells are compared directly with no margin, so a zero at the same shell as the masked 0.143 resolution passes and a zero at a finer shell fails.
2. The masked FSC itself reaches 0 or below at some shell beyond its 0.143 crossing, at or before Nyquist. There is no exception for maps whose 0.143 crossing lies near Nyquist.

Crossings are shell values and are not interpolated. The 0.143 resolution is the resolution of the last shell at or above 0.143 before the first shell below it. The phase randomized zero is the resolution of the first shell with a value at or below 0. If the masked FSC never drops below 0.143, the phase randomized FSC never reaches 0, or the masked FSC never reaches 0 after its 0.143 drop, the mask fails.

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

The pass verdict (`prfsc_pass`) is built from three quantities:

1. **Masked FSC at 0.143**: the resolution of the last shell at or above 0.143 before the first shell below it (`masked_res_0_143`).
2. **Phase randomized FSC zero**: the resolution of the first shell with phase randomized FSC at or below 0 (`phase_rand_zero_res`).
3. **Masked FSC zero**: the resolution of the first shell, at or after the first shell below 0.143, where the masked FSC is at or below 0 (`masked_zero_res`, NaN if there is none).

`prfsc_pass` is true when `phase_rand_zero_res` is at or coarser than `masked_res_0_143` (larger or equal in Å) and `masked_zero_res` exists.

The unmasked 0.5 resolution, the corrected 0.143 resolution, the noise floor and the correction magnitude are still reported. They are not part of the verdict. The older rule (phase randomized zero at or coarser than the unmasked 0.5 resolution) is reported as `legacy_criterion_met` and does not decide pass.

## Interpretation of Results

### CSV Output

The `mask3d_evaluation.csv` file contains:
- `prfsc_pass`: Boolean indicating if the mask passes the PRFSC criteria
- `masked_res_0_143`: Resolution at FSC=0.143 of the masked FSC (Å)
- `phase_rand_zero_res`: Resolution at phase randomized FSC zero crossing (Å)
- `masked_zero_res`: Resolution where the masked FSC reaches 0 beyond its 0.143 crossing (Å, NaN if it does not)
- `legacy_criterion_met`: Reported only. Boolean for the older rule (phase randomized zero at or coarser than the unmasked 0.5 resolution)
- `unmasked_res_0_5`: Resolution at FSC=0.5 without mask (Å), reported only
- `corrected_res_0_143`: Resolution at FSC=0.143 with corrected FSC (Å), reported only
- `correction_magnitude` and `phase_rand_noise_floor`: reported only
- `valid`: Boolean indicating if the unmasked 0.5, phase randomized zero and corrected 0.143 crossings were found
- `valid_masked_0_143`, `valid_phase_rand_zero` and `valid_masked_zero`: whether each crossing used by `prfsc_pass` was found

### Plot Output

The FSC curves plot shows:
- **Blue line**: Unmasked FSC
- **Orange line**: Masked FSC
- **Green line**: Phase Randomized FSC
- **Red line**: Corrected FSC
- **Vertical lines**: Key resolution thresholds
- **Annotations**: Resolution values at important thresholds

### Summary Star File

The `mask_evaluation_summary.star` file contains the pass verdict, the three resolutions behind it and the reported-only quantities that can be used in downstream processing or analysis.