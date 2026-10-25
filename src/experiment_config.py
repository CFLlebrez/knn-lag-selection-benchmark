"""Typed experiment configuration for the forecasting benchmark.

The legacy command-line interface represents forecast intent through a single
``fh`` integer.  These dataclasses make the intended targets, subset scope,
temporal validation, selector, and evaluator explicit without changing the
existing execution path yet.
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence


SubsetScope = Literal["shared", "per_horizon"]


@dataclass(frozen=True)
class ForecastSpec:
    """Defines which target timestamps an experiment predicts.

    Step 0 means the target at the forecast origin. Positive values represent
    future offsets. Target lags remain eligible predictors independently of
    this setting.
    """

    steps: tuple[int, ...] = (0,)
    subset_scope: SubsetScope = "shared"
    include_current_target: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(self.steps)
        if not normalized:
            raise ValueError("Forecast steps cannot be empty")
        if any(not isinstance(step, int) for step in normalized):
            raise TypeError("Forecast steps must be integers")
        if any(step < 0 for step in normalized):
            raise ValueError("Forecast steps cannot be negative")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Forecast steps must be unique")
        if tuple(sorted(normalized)) != normalized:
            raise ValueError("Forecast steps must be sorted")
        if self.subset_scope not in ("shared", "per_horizon"):
            raise ValueError(f"Unknown subset scope: {self.subset_scope}")
        if 0 in normalized and self.include_current_target:
            raise ValueError(
                "The current target cannot be both a predictor and a step-0 label"
            )
        object.__setattr__(self, "steps", normalized)

    @classmethod
    def immediate(cls, **kwargs) -> "ForecastSpec":
        return cls(steps=(0,), **kwargs)

    @classmethod
    def direct(cls, horizon: int, **kwargs) -> "ForecastSpec":
        if horizon <= 0:
            raise ValueError("A direct forecast horizon must be positive")
        return cls(steps=(horizon,), **kwargs)

    @classmethod
    def path(cls, horizon: int, **kwargs) -> "ForecastSpec":
        if horizon <= 0:
            raise ValueError("A forecast path horizon must be positive")
        return cls(steps=tuple(range(1, horizon + 1)), **kwargs)


@dataclass(frozen=True)
class SplitSpec:
    """Chronological hold-out and development-CV configuration."""

    test_fraction: float = 0.20
    cv_splits: int = 3
    purge_forecast_overlap: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    n_features: int
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Selector name cannot be empty")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")


@dataclass(frozen=True)
class EvaluatorSpec:
    name: str = "knn"
    tune: bool = True
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Evaluator name cannot be empty")


@dataclass(frozen=True)
class ExperimentConfig:
    forecast: ForecastSpec
    split: SplitSpec
    selector: SelectorSpec
    evaluator: EvaluatorSpec = field(default_factory=EvaluatorSpec)
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[int],
        selector: SelectorSpec,
        **kwargs,
    ) -> "ExperimentConfig":
        return cls(
            forecast=ForecastSpec(steps=tuple(steps)),
            split=kwargs.pop("split", SplitSpec()),
            selector=selector,
            **kwargs,
        )
