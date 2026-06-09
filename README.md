# Confidence Model Research Repository

This repository contains the modeling and validation workflow for dynamic
confidence models in human-AI decision making.

The project has three historical layers:

1. Legacy chess study materials from the original project.
2. A newer confidence dynamics model developed on the chess data.
3. Truss-study validation and model improvement using the older truss dataset.

The current research goal is to test whether the improved model developed on
the chess data generalizes to the truss data, and to use the richer truss action
records to refine mechanisms such as modification, alignment, and confidence
coupling.

## Main Areas

- `Data/` and `Code/`: legacy chess data and MATLAB code from the original
  project.
- `legacy_chess/`: historical papers and notes for the original chess study.
- `chess_model/`: papers and documentation for the improved chess-derived
  confidence model.
- `benchmark.py` and `original_common.py`: Python reproduction of the original
  Chong-style confidence model on the chess data.
- `Paper_fig/`: model variants, paper figures, and chess-model outputs.
- `truss_data/`: truss dataset, reconstructed detailed action records, baseline
  benchmarks, detailed model fits, and validation outputs.
- `truss_validation/`: documentation stub for the current truss validation
  research line.
- `archive/`: old notes and extracted text artifacts.
- `outputs_extended/`: earlier extended-model outputs.

## Recommended Starting Points

- `docs/project_map.md`: conceptual map of the repository.
- `docs/runbook.md`: commands for reproducing the main analyses.
- `docs/truss_validation.md`: current truss-validation pipeline and key result
  files.

## Current Key Result

The truss validation currently supports the improved dynamic model:

- `truss_data/data_folder_output/detailed_available_baseline_full_metrics.csv`
- `truss_data/data_folder_output/detailed_available_robust_summary.csv`

In these outputs, the detailed physics model (`PHYS-FULL-DETAILED`) improves
over the original baseline (`BASELINE-OG`) on truss condition-mean fit metrics.
