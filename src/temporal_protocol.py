"""Leakage-resistant temporal splitting utilities."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .experiment_config import ForecastSpec, SplitSpec


@dataclass(frozen=True)
class TemporalHoldout:
    """Development and final-test partitions plus their positional boundary."""

    development: pd.DataFrame
    test: pd.DataFrame
    test_start: int
    purged_rows: int


def make_temporal_holdout(
    frame: pd.DataFrame,
    forecast: ForecastSpec,
    split: SplitSpec,
) -> TemporalHoldout:
    """Create a chronological hold-out without crossing forecast labels.

    The test boundary is based on forecast-origin rows. When a positive forecast
    step is requested, the last development origins are purged so none of their
    labels reach the first test origin. Test rows may still use legitimate past
    context supplied by the transformed feature columns.
    """

    if frame.empty:
        raise ValueError("Cannot split an empty dataset")

    test_start = int(len(frame) * (1.0 - split.test_fraction))
    if test_start <= 0 or test_start >= len(frame):
        raise ValueError("The requested test fraction leaves an empty partition")

    max_step = max(forecast.steps)
    purge = max_step if split.purge_forecast_overlap else 0
    development_end = test_start - purge
    if development_end <= 0:
        raise ValueError(
            "Forecast-horizon purging leaves no development observations"
        )

    development = frame.iloc[:development_end].copy()
    test = frame.iloc[test_start:].copy()
    return TemporalHoldout(
        development=development,
        test=test,
        test_start=test_start,
        purged_rows=purge,
    )


def make_development_cv(
    forecast: ForecastSpec,
    split: SplitSpec,
) -> TimeSeriesSplit:
    """Build expanding-window CV with a gap protecting future labels."""

    gap = max(forecast.steps) if split.purge_forecast_overlap else 0
    return TimeSeriesSplit(n_splits=split.cv_splits, gap=gap)
