# Truss Analysis Notes

This document records confirmed truss-analysis findings only. Exploratory ideas
and speculative mechanisms should stay in separate working notes until they are
validated.

## Modify Feedback And Confidence Updates

### Question

When a participant modifies rather than simply accepting or rejecting the AI
suggestion, does the feedback on that modify trial update AI-confidence and
self-confidence differently?

### Data

Source file:

- `truss_data/modify_analysis/trial_level_modify_analysis.csv`

Analysis outputs:

- `truss_data/modify_analysis/modify_feedback_confidence_delta_summary.csv`
- `truss_data/modify_analysis/modify_feedback_confidence_delta_tests.csv`

The analysis uses modify trials only:

- negative-feedback modify trials: 418 trials from 95 participants
- positive-feedback modify trials: 203 trials from 75 participants

Confidence deltas are computed as:

```text
delta_ai = AI confidence after the trial - AI confidence before the trial
delta_self = self-confidence after the trial - self-confidence before the trial
```

### Descriptive Results

Negative-feedback modify trials:

- mean delta_ai: -0.039
- mean delta_self: -0.059
- mean delta_self_minus_ai: -0.020

Positive-feedback modify trials:

- mean delta_ai: +0.038
- mean delta_self: +0.080
- mean delta_self_minus_ai: +0.042

### Statistical Tests

Positive vs negative modify feedback significantly predicts confidence-update
direction in both channels.

AI-confidence:

- trial-level Welch test difference: +0.078, p = 4.33e-05
- participant-paired difference: +0.094, p = 0.00194
- clustered OLS controlling condition and trial: +0.077, p = 5.16e-05

Self-confidence:

- trial-level Welch test difference: +0.139, p = 2.24e-14
- participant-paired difference: +0.200, p = 1.54e-09
- clustered OLS controlling condition and trial: +0.140, p = 2.94e-10

Self-minus-AI confidence gap:

- trial-level Welch test difference: +0.062, p = 0.00543
- participant-paired difference: +0.106, p = 0.00888
- clustered OLS controlling condition and trial: +0.063, p = 0.00849

### Summary Observation

Modify feedback updates both confidence channels, but the feedback effect is
larger for self-confidence than for AI-confidence.

### Conclusion

For truss modify trials, feedback should be modeled as a confidence-update
event rather than only as a performance outcome. Positive modify feedback
increases both AI-confidence and self-confidence relative to negative modify
feedback, with a stronger and more reliable effect on self-confidence.

