# Project Map

This repository is organized around one research question:

Can a dynamic confidence model, first improved on the chess human-AI dataset,
generalize to the older truss human-AI dataset and explain additional action
mechanisms such as modification and alignment?

## Research Lineage

### 1. Legacy chess materials

These are the historical anchor materials from the original project.

- `Data/`: chess participant CSV files.
- `Code/`: original MATLAB data and scripts.
- `original_model.pdf`: original confidence-model paper.
- `data_paper.pdf`: paper associated with the original data.
- `comparison_analysis.md`: older comparison notes. The file appears to have
  mojibake/encoding issues but still documents the original-vs-extended model
  contrast.

### 2. Python reproduction of the original model

These files reconstruct the original model and produce baseline outputs.

- `benchmark.py`: complete Python reproduction of the original Chong-style
  analysis on the chess data.
- `original_common.py`: shared utilities for parsing chess data and simulating
  the original 2-action, 4-experience model.
- `original_align_analysis.py` and `original_modify_analysis.py`: diagnostics
  around alignment and modification under the original model framing.

The original model uses four experience channels:

- accept AI, positive feedback
- reject AI, positive feedback
- accept AI, negative feedback
- reject AI, negative feedback

The main update equation is a weighted evidence and memory process for AI
confidence and self-confidence.

### 3. Improved chess model

This is the modeling contribution developed from the chess data and associated
with `my_paper.pdf`.

- `my_paper.pdf`: conference paper based on the improved model.
- `Paper_fig/full_model.py`: main physics-style confidence dynamics model.
- `Paper_fig/og_model.py`: original model comparison inside the figure pipeline.
- `Paper_fig/physics_*.py`, `Paper_fig/phys_noB.py`, `Paper_fig/og_physics.py`:
  ablations and alternative dynamics.
- `Paper_fig/manu_figure.py`: manuscript figure assembly.
- `Paper_fig/*_output/`: saved model parameters, robustness files, and plots.

The improved model frames confidence as a dynamic system:

```text
evidence -> memory -> force -> velocity -> confidence
```

It also treats modify and alignment mechanisms as meaningful model components,
rather than only collapsing behavior into accept/reject.

### 4. Truss validation and improvement

This is the current forward-looking part of the project.

- `truss_data/`: truss dataset and all validation outputs.
- `truss_data/truss_og_benchmark.py`: original baseline model on truss data.
- `truss_data/truss_detailed_subset_fit.py`: baseline vs improved/detailed
  physics model on participants with reconstructed detailed action records.
- `truss_data/truss_detailed_component_table.py`: component-level and ablation
  comparisons.
- `truss_data/modify_analysis/`: modify-trial analysis and derived labels.
- `truss_data/data_folder_output/`: main detailed truss outputs.
- `truss_data/og_benchmark_output/`: original baseline outputs on truss data.

Current truss results support using the older truss data as an external
validation dataset for the improved confidence model.

## Important Path Caution

Many scripts use hard-coded relative paths such as `Data`, `Paper_fig`, and
`truss_data/data_folder_output`. For that reason, this repository has not yet
been physically reorganized into new folders. The current cleanup is
documentation-first so the analyses remain reproducible.

If files are moved later, update paths in the scripts first or introduce a
central path configuration.

