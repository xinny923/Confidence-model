# Truss Validation

This area marks the current forward-looking research direction: validating and
improving the chess-derived confidence model on the older truss dataset.

## Current Physical Location

The truss data and scripts are still stored in:

- `truss_data/`

That folder is intentionally not moved yet because truss scripts refer to paths
inside `truss_data/` directly.

## Key Pipeline

- `truss_data/truss_og_benchmark.py`: original baseline on truss data.
- `truss_data/truss_detailed_subset_fit.py`: detailed truss validation of the
  improved model.
- `truss_data/modify_analysis/`: modify-trial analysis.
- `truss_data/data_folder_output/`: current detailed validation outputs.

See `docs/truss_validation.md` for the current model-comparison result and
next analysis ideas.

