# Truss Modify Analysis

This folder analyzes modify trials in the complete truss detailed data.

Modify labels are deliberately organized as three paired dimensions rather than one mutually exclusive category:

- outcome: productive_modify vs unproductive_modify
- agency: self_correction vs partial_acceptance, with agency_unclear when neither definition fires
- coupling: collaborative_refinement vs conflict_modify, with coupling_unclear when neither definition fires

Operational definitions:

- productive_modify: modify trial with positive feedback.
- unproductive_modify: modify trial with negative feedback.
- self_correction: modify trial where self-confidence increases and AI-confidence does not increase.
- partial_acceptance: modify trial where AI-confidence increases and self-confidence does not decrease.
- collaborative_refinement: productive modify trial where both AI-confidence and self-confidence increase.
- conflict_modify: modify trial where AI-confidence and self-confidence move in opposite directions.

Confidence deltas use C[t+1] - C[t] for each trial.
