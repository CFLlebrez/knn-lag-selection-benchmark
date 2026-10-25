# Release lineage

## CAEPIA 2026 paper snapshot (`Milestone`)

This tag records the implementation associated with the camera-ready paper:
"A k-NN-Based Benchmark for Lag Selection in Time Series Forecasting."

The snapshot is retained for traceability. It must not be confused with the
corrected experimental protocol described below.

## Protocol-corrected release (`v0.2.0`)

This release establishes the development protocol for subsequent experiments:

- direct-horizon target semantics for `--fh H`;
- target lags remain eligible while current and future target labels are
  excluded from predictors;
- horizon-aware purging at the development/test boundary;
- expanding-window selector and evaluator tuning on development data only;
- temporal internal validation for Lasso, Elastic Net, and sequential
  selection;
- a shared leakage-safe candidate feature space for the baseline and selected
  subset; and
- explicit tests and documentation for the two-stage tuning procedure.

Because these changes correct the evaluation protocol, results produced by
`v0.2.0` are new experiments and should not be presented as the unchanged
camera-ready results.
