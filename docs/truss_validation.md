# Truss Validation Notes

The truss dataset is the next validation target for the improved confidence
model. It is useful because it was collected in a different task domain while
still preserving human-AI confidence, feedback, and action traces.

## Research Purpose

The current goal is:

1. Reproduce the original confidence model on truss data.
2. Test whether the improved chess-derived dynamics generalize to truss.
3. Use detailed truss action traces to improve the model through modify and
   alignment mechanisms.

## Data Inputs

Important truss inputs:

- `truss_data/C_data.mat`: AI-confidence matrix by condition.
- `truss_data/sC_data.mat`: self-confidence matrix by condition.
- `truss_data/e_data.mat`: original 4-channel experience tensor.
- `truss_data/act_data.mat`: original action matrix.
- `truss_data/feed1_data.mat`, `truss_data/feed2_data.mat`: feedback matrices.
- `truss_data/score1_data.mat`, `truss_data/score2_data.mat`: score matrices.
- `truss_data/data/.P#/data#.csv`: participant-level confidence files used by
  the detailed reconstruction pipeline.
- `truss_data/data_folder_output/data_folder_reconstructed_trials.csv`:
  reconstructed detailed trial table.

## Baseline Pipeline

Script:

- `truss_data/truss_og_benchmark.py`

Purpose:

- Fit the original 8-parameter confidence model to truss AI confidence and
  self-confidence.
- Evaluate condition-mean MSE, R2, and adjusted R2.
- Save participant summaries, trial-level fit, and robustness outputs.

Main output folder:

- `truss_data/og_benchmark_output/`

## Detailed Pipeline

Script:

- `truss_data/truss_detailed_subset_fit.py`

Purpose:

- Compare `BASELINE-OG` against physics-style variants built from detailed
  reconstructed truss actions.
- Test component contributions from modify handling and alignment.
- Bootstrap or subset-sample robustness of the model comparison.

Model labels:

- `BASELINE-OG`: original truss benchmark model.
- `PHYS-FULL-NO-MOD-NO-ALIGN`: physics dynamics without detailed modify or
  alignment mechanisms.
- `PHYS-FULL+MOD`: physics dynamics with modify handling.
- `PHYS-FULL+ALIGN`: physics dynamics with alignment handling.
- `PHYS-FULL-DETAILED`: full detailed model with both modify and alignment.

Main output files:

- `truss_data/data_folder_output/detailed_available_baseline_full_metrics.csv`
- `truss_data/data_folder_output/detailed_available_baseline_full_params.csv`
- `truss_data/data_folder_output/detailed_available_robust_raw.csv`
- `truss_data/data_folder_output/detailed_available_robust_summary.csv`

## Current Key Result

The detailed truss validation currently favors the full detailed model.

From `detailed_available_baseline_full_metrics.csv`:

- `BASELINE-OG` mean adjusted R2 is about 0.748.
- `PHYS-FULL-DETAILED` mean adjusted R2 is about 0.842.

From `detailed_available_robust_summary.csv`:

- `BASELINE-OG` mean adjusted R2 is about 0.721.
- `PHYS-FULL-DETAILED` mean adjusted R2 is about 0.809.

This supports the argument that the improved confidence dynamics generalize
from chess to truss and that detailed action mechanisms add explanatory power.

## Next Analysis Ideas

- Quantify which component contributes more across participants: modify,
  alignment, or their interaction.
- Compare chess and truss parameter distributions directly.
- Separate high-performing and low-performing participants in truss and test
  whether confidence dynamics differ.
- Turn the truss detailed results into one clean validation figure and one
  model-comparison table.

