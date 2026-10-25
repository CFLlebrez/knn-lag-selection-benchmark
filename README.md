# $k$-NN-Based Feature Selection Comparative Framework

## Overview
This framework provides an end-to-end automated pipeline designed for high-dimensional time series forecasting. It integrates dynamic temporal data transformation, sophisticated feature selection architectures, and predictive validation. By bridging the gap between classical statistical selection methods (inspired by the Weka workbench) and modern machine learning workflows, the tool enables researchers to identify optimal feature subsets while mitigating the curse of dimensionality common in lagged time series datasets.

---

## Core Pipeline Architecture

The framework operates through three distinct stages to ensure a rigorous experimental flow:

1. **Temporal Transformation**: Converts univariate or multivariate time series into a supervised learning format by generating lagged observations ($t-n$) based on a user-defined Past History ($ph$) and Forecast Horizon ($fh$).
2. **Feature Selection (FS)**: Executes a two-phase selection process. First, a **Tuning Phase** optimizes the hyperparameters of the chosen selection method (e.g., regularization strength in Lasso or directionality in Sequential Selection). Second, the **Selection Phase** identifies the most relevant features and generates metadata including importance scores.
3. **Predictive Evaluation**: Validates the selected subset using an evaluation regressor. The framework automatically tunes the internal hyperparameters (such as the $k$ parameter for KNN) for both the full feature set (baseline) and the reduced subset to provide a fair performance comparison.

---

## Technical Specifications and Parameters

### Execution Command
The framework is executed via `main.py` using the following standardized syntax:
```bash
python main.py [input_file] [output_dir] --fv [index] --fh [value] --ph [value] --time_col [name] [options]
```
### Primary Arguments

#### Target & Temporal Configuration
* **`--fv` (int)**: Forecast Variable index specifying the target column in the input dataset.
* **`--fh` (int)**: Forecast Horizon indicating the number of future temporal steps to predict.
* **`--ph` (int)**: Past History defining the look-back window size for generating lagged features.
* **`--time_col` (str)**: Name of the timestamp column to ensure correct chronological alignment during data transformation and temporal splitting.

#### Feature Selection Module
* **`--feature_selection` (str)**: Enables the reduction module and specifies the feature selection algorithm to deploy (e.g., `ccf`, `cfs`, `sequential`).
* **`--fs_n_features` (int)**: Target number of optimal features to retain after the selection phase.

#### Evaluation & Predictive Benchmarking
* **`--evaluation`**: Explicit flag required to trigger the final predictive benchmarking and validation stage. If omitted, the pipeline halts after exporting selection metadata.
* **`--evaluation_model` (str)**: Specifies the downstream machine learning architecture used to evaluate the subsets. Options include:
    * `knn`: K-Nearest Neighbors Regressor (includes automated Bayesian tuning via Optuna for optimal $k$ selection).
    * `mlp`: Multi-Layer Perceptron Regressor (Neutral architectural mode).
    * `hgb`: HistGradientBoosting Regressor (Neutral architectural mode).

#### Reproducibility & Tracking
* **`--seed` (int)**: Sets the global random seed for `numpy`, `random`, and environment variables, guaranteeing deterministic behavior across multi-seed experimental runs.
* **`--run_name` (str)**: Identifier for the current execution, used to label output directories and isolate master comparative reports.

---

### Feature Selection Taxonomy
The framework categorizes methods based on their underlying mathematical approach:

| Category | Methods | Description |
| :--- | :--- | :--- |
| **Linear / Statistical** | `pearson`, `ccf`, `lasso`, `elastic_net` | Evaluates linear dependencies and employs $L_1/L_2$ regularization. |
| **Information Theory** | `mutual_info` | Captures non-linear dependencies via entropy-based metrics. |
| **Wrapper / Iterative** | `rfe`, `sequential` | Uses model-in-the-loop selection (Support Vector Regression or Linear Regression). |
| **Ensemble** | `random_forest` | Leverages Gini impurity or permutation importance. |
| **Heuristic (Weka-Inspired)** | `cfs` | Correlation-based Feature Selection that penalizes feature redundancy. |
| **Statistical Filters** | `sklearn_filter` | Univariate tests including `selectkbest` and `selectpercentile`. |

---

### Academic Novelty and Contributions
The framework introduces several key contributions to the field of automated machine learning (AutoML) for time series:
* **$k$-NN as a Diagnostic Evaluator**: Unlike workflows that couple feature selection with highly parameterized, black-box forecasting models, this framework leverages a non-parametric $k$-Nearest Neighbors ($k$-NN) regressor as a core diagnostic tool.
* **Optimized Weka-to-Python Port**: Specifically, the implementation of Correlation-based Feature Selection (CFS) has been optimized for temporal data, utilizing Pearson correlation matrices to maximize feature-to-target relevance while minimizing feature-to-feature redundancy.
* **Hybrid Hyperparameter Tuning**: Unlike standard selection libraries, this framework performs an internal optimization of the selection method's parameters (e.g., `alpha` for Lasso) before the evaluation of the predictive model.
* **Dynamic Lag Handling**: The system manages the high dimensionality resulting from large window sizes ($ph$), automatically filtering out non-contributory lags that often introduce noise into forecasting models.

---

### Experimental Outputs
Upon completion, the framework generates a structured directory containing:
* **Transformed Dataset**: The processed CSV containing all generated lags.
* **Selection Metadata**: A JSON file detailing the selected feature indices, execution time, and importance rankings.
* **Evaluation Report**: A comparative analysis featuring Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared ($R^2$) metrics and various times (Selection, Transformation and Inference) for both the baseline and the optimized model (after feature selection using different methods).
