# Runbook

This runbook lists the main reproducible analysis commands. Commands should be
run from the repository root unless noted otherwise.

## Environment

Recommended Python packages:

```powershell
python -m pip install -r requirements.txt
```

If Excel export is needed, install `openpyxl` as well. Most scripts fall back to
CSV if Excel support is unavailable.

## Main Chess Baseline

Reproduce the original chess confidence-model benchmark:

```powershell
python benchmark.py
```

Main outputs are written to `output_benchmark/` when generated. This directory
is not currently committed in the repository, but some downstream scripts refer
to its parameter table.

## Improved Chess Model

Run the main physics-style confidence model used for the improved chess-model
line:

```powershell
python Paper_fig\full_model.py
```

Main outputs:

- `Paper_fig/full_output/full_model_params.csv`
- `Paper_fig/full_output/robust_metrics.csv`
- `Paper_fig/full_output/robustness_statistics.csv`
- `Paper_fig/full_output/figure_conf.png`

Related model variants:

```powershell
python Paper_fig\og_model.py
python Paper_fig\og_physics.py
python Paper_fig\physics_A.py
python Paper_fig\physics_gain.py
python Paper_fig\physics_res.py
python Paper_fig\physics_assym.py
python Paper_fig\phys_noB.py
```

## Truss Original Baseline

Run the original confidence model on the truss dataset:

```powershell
python truss_data\truss_og_benchmark.py
```

Useful options:

```powershell
python truss_data\truss_og_benchmark.py --no-robustness
python truss_data\truss_og_benchmark.py --robustness-iterations 20
```

Main outputs:

- `truss_data/og_benchmark_output/og_model_params.csv`
- `truss_data/og_benchmark_output/participant_summary.csv`
- `truss_data/og_benchmark_output/robust_metrics.csv`
- `truss_data/og_benchmark_output/robustness_statistics.csv`
- `truss_data/og_benchmark_output/figure_condition_fit.png`

## Truss Detailed Validation

Run the current main truss validation pipeline:

```powershell
python truss_data\truss_detailed_subset_fit.py
```

Main outputs:

- `truss_data/data_folder_output/detailed_available_baseline_full_metrics.csv`
- `truss_data/data_folder_output/detailed_available_baseline_full_params.csv`
- `truss_data/data_folder_output/detailed_available_metadata.csv`
- `truss_data/data_folder_output/detailed_available_robust_raw.csv`
- `truss_data/data_folder_output/detailed_available_robust_summary.csv`

This script runs robustness iterations by default and may take time.

## Truss Modify Analysis

The modify-trial analysis is documented in:

- `truss_data/modify_analysis/README.md`

Run:

```powershell
python truss_data\modify_analysis\run_modify_analysis.py
```

Main outputs:

- `truss_data/modify_analysis/action_feedback_delta_summary.csv`
- `truss_data/modify_analysis/participant_modify_summary.csv`
- `truss_data/modify_analysis/performer_group_summary.csv`

## Known Caveats

- Some older scripts refer to modules that are not present in the current
  repository, such as `mdiscrete_core.py`. Treat those as historical or
  partially migrated scripts.
- Several scripts assume specific relative paths and should be run from the
  repository root.
- The repository contains many saved `.mat` and `.png` experiment artifacts.
  Re-running all analysis can be slow and may overwrite existing outputs.

