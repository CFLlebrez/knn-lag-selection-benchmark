# k-NN-Based Feature-Selection Benchmark for Time Series

This repository compares feature-selection methods on lagged time-series data. A
simple downstream regressor acts as a diagnostic arbiter: the same model family
is evaluated with the complete eligible feature set and with each selected
subset.

## Forecasting contract

For a target named `y`:

- `--fh 0` predicts `y` at the forecast origin.
- `--fh H`, for `H > 0`, is a direct forecast of `y_t+H`.
- Target lags such as `y_t-1` are valid candidate predictors.
- The current target `y` and every future target column `y_t+*` are excluded
  from the candidate predictors.

The transformer materializes the intermediate future columns up to `H`, but
only `y_t+H` is used as the label. Forecast-path and per-horizon subset modes
are represented in the internal configuration model but are not exposed by the
current command-line interface.

## Experimental protocol

Each run uses a chronological 80/20 development/test hold-out. For horizon
`H`, the last `H` development origins are purged so their labels cannot cross
the test boundary. The final test period is not used for feature selection or
hyperparameter tuning.

Within the development period:

1. Selector hyperparameters, when present, are assessed with expanding-window
   cross-validation. Feature selection is refitted inside every training fold.
2. Lasso, Elastic Net, and sequential selection use temporal internal
   cross-validation rather than ordinary future-looking folds.
3. The downstream model is tuned separately for the full candidate feature set
   and the selected subset, using the same expanding-window protocol.
4. The selector is fitted once on all development data.
5. The baseline and selected-subset models are fitted on development data and
   evaluated once on the untouched test period.

Every cross-validation boundary uses a gap of `H` rows for a direct forecast at
horizon `H`. With `--fh 0`, no gap is required.

## Installation

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On macOS or Linux, use `.venv/bin/python` instead.

## Running an experiment

```bash
python main.py INPUT.csv OUTPUT_NAME \
  --fv TARGET_INDEX \
  --fh HORIZON \
  --ph PAST_HISTORY \
  --time_col TIMESTAMP_COLUMN \
  --feature_selection \
  --fs_method pearson \
  --fs_n_features 10 \
  --evaluation \
  --evaluation_model knn \
  --seed 42 \
  --run_name example
```

Relative input paths are resolved below `input_csv_files/`. Results are written
below `results/OUTPUT_NAME/`. Absolute paths may also be supplied.

Important arguments:

- `--fv`: zero-based target-column index in the original CSV.
- `--fh`: non-negative direct forecast horizon.
- `--ph`: non-negative number of lag steps.
- `--time_col`: timestamp column used to sort observations chronologically.
- `--feature_selection`: enables feature selection.
- `--fs_method`: selector name.
- `--fs_n_features`: positive subset cardinality; required when selection is
  enabled.
- `--evaluation`: evaluates the baseline and selected subset; requires feature
  selection.
- `--evaluation_model`: `knn`, `mlp`, or `hgb`.
- `--seed`: seed used by stochastic selectors, evaluators, and Optuna.

Run `python main.py --help` for the complete option list.

## Available selectors

| Family | Methods |
| --- | --- |
| Correlation and information | `pearson`, `ccf`, `mutual_info` |
| Embedded/model based | `random_forest`, `lasso`, `elastic_net` |
| Wrapper | `rfe`, `sequential` |
| scikit-learn filters | `sklearn_filter` |
| Weka-inspired | `weka_inspired` (CFS) |

External Optuna tuning is currently meaningful for Random Forest and mutual
information. Lasso and Elastic Net choose their regularization through their
own temporal cross-validation unless a fixed `--alpha` is supplied. Methods
without a genuine external search space skip the Optuna stage rather than
reporting a no-op optimization.

## Outputs

A completed run produces:

- the lagged supervised dataset;
- feature-importance reports and selection metadata;
- the exact leakage-safe candidate feature list used by the baseline;
- the selected feature list and direct target column;
- MAE, RMSE, R-squared, training time, and inference time for the baseline and
  selected subset; and
- a comparative CSV for the chosen downstream model.

The transformed dataset may be cached across runs. Run-specific reports are
placed in a seed-labelled directory when `--seed` is provided.

## Tests

```bash
.venv/Scripts/python -B -m unittest discover -s tests -p "test_*.py" -v
```

The tests cover target alignment, chronological sorting, purged temporal
splits, selector isolation inside development folds, direct-horizon target
semantics, leakage-safe baseline construction, parameter propagation, and an
end-to-end CLI run.

## Published and continuing versions

The conference implementation should be preserved with an immutable Git tag
and, when appropriate, an archived release linked from the paper. Development
can continue on branches and later releases without rewriting that historical
state. Results produced after protocol corrections must be identified as new
experiments and should not be presented as the unchanged camera-ready results.
