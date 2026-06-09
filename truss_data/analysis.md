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

## Non-Monotonic Confidence Recalibration

### Question

Do opposite-direction confidence updates reflect relative recalibration between
AI-confidence and self-confidence?

Opposite-direction, or split, updates are defined as:

```text
AI_up_self_down: AI-confidence increases while self-confidence decreases
AI_down_self_up: AI-confidence decreases while self-confidence increases
```

Monotonic nonzero updates are defined as:

```text
AI_up_self_up
AI_down_self_down
```

### Data

Source file:

- `truss_data/modify_analysis/trial_level_modify_analysis.csv`

Analysis outputs:

- `truss_data/modify_analysis/conjugate_logistic_bootstrap_summary.csv`
- `truss_data/modify_analysis/split_vs_monotonic_abs_gap_test.csv`
- `truss_data/modify_analysis/modify_vs_nonmodify_coupling_bootstrap_summary.csv`

The main analyses use nonzero confidence-update trials only:

- split updates: 268 trials
- monotonic updates: 411 trials

### Prior Imbalance

Split updates occurred when the prior AI/self confidence imbalance was much
larger than in monotonic updates.

```text
mean abs(self_before - ai_before)

split updates:    0.395
monotonic updates: 0.111
difference:       0.283
```

Tests:

- Welch test: p = 5.53e-44
- participant-cluster bootstrap CI for the difference: [0.243, 0.323]
- participant-cluster bootstrap p = 0.00664

### Direction Of Split Updates

Among split updates, the prior self-minus-AI confidence gap strongly predicted
which direction the split took.

Model:

```text
AI_up_self_down vs AI_down_self_up
```

Key predictor:

```text
prior_gap = self_before - ai_before
```

Result:

- coefficient for prior_gap: +21.91
- bootstrap CI: [19.16, 33.99]
- bootstrap p = 0.00664

This means positive prior gap strongly predicts `AI_up_self_down`, while
negative prior gap predicts `AI_down_self_up`.

### Reject-Negative Concentration

`AI_up_self_down` updates were also associated with the combination of rejecting
the AI and receiving negative feedback.

Model:

```text
AI_up_self_down vs other nonzero confidence updates
```

Key result:

- reject x negative feedback coefficient: +4.45
- bootstrap CI: [1.72, 9.36]
- bootstrap p = 0.00664

The prior gap remained strongly positive in the same model:

- prior_gap coefficient: +10.62
- bootstrap CI: [8.28, 16.48]
- bootstrap p = 0.00664

### Modify Coupling

AI/self confidence deltas were more strongly coupled in modify trials than in
non-modify trials.

```text
Pearson r(delta_ai, delta_self)

modify trials:     0.297
non-modify trials: 0.060
difference:        0.238
```

Participant-cluster bootstrap:

- difference in r CI: [0.082, 0.416]
- difference in Fisher z CI: [0.085, 0.460]
- bootstrap p = 0.00664

### Summary Observation

Split confidence updates are associated with larger prior imbalance between
self-confidence and AI-confidence. The sign of the prior gap predicts the
direction of the split update, and modify trials show stronger coupling between
the two confidence channels than non-modify trials.

### Conclusion

Non-monotonic confidence updates support a relative recalibration interpretation:
when one confidence channel is much higher than the other, subsequent feedback
can shift confidence away from the initially dominant channel and toward the
other channel. This effect is strongest and cleanest for the
`AI_up_self_down` pattern following reject-and-negative-feedback events.
