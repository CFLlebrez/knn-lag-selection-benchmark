import unittest

from src.experiment_config import (
    EvaluatorSpec,
    ExperimentConfig,
    ForecastSpec,
    SelectorSpec,
    SplitSpec,
)


class ForecastSpecTests(unittest.TestCase):
    def test_immediate_direct_and_path_constructors(self):
        self.assertEqual(ForecastSpec.immediate().steps, (0,))
        self.assertEqual(ForecastSpec.direct(6).steps, (6,))
        self.assertEqual(ForecastSpec.path(3).steps, (1, 2, 3))

    def test_rejects_current_target_as_step_zero_predictor(self):
        with self.assertRaises(ValueError):
            ForecastSpec(steps=(0,), include_current_target=True)

    def test_rejects_duplicate_or_unsorted_steps(self):
        with self.assertRaises(ValueError):
            ForecastSpec(steps=(1, 1))
        with self.assertRaises(ValueError):
            ForecastSpec(steps=(2, 1))


class ExperimentConfigTests(unittest.TestCase):
    def test_complete_configuration(self):
        config = ExperimentConfig(
            forecast=ForecastSpec.path(24),
            split=SplitSpec(test_fraction=0.2, cv_splits=3),
            selector=SelectorSpec(name="mutual_info", n_features=6),
            evaluator=EvaluatorSpec(name="knn"),
            seed=123,
        )

        self.assertEqual(config.forecast.steps[0], 1)
        self.assertEqual(config.forecast.steps[-1], 24)
        self.assertEqual(config.selector.n_features, 6)
        self.assertEqual(config.seed, 123)


if __name__ == "__main__":
    unittest.main()
