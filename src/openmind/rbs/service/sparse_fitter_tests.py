import math

import numpy as np
import pytest
from scipy.special import expit

from openmind.rbs.service.sparse_fitter import SparseFitter


def synthetic() -> tuple[np.ndarray, np.ndarray]:
    """Five standard normal columns; the targets follow a logistic model reading only the first and the fourth."""
    columns = np.random.default_rng(1).normal(size=(2000, 5))
    return columns, expit(columns @ np.array([2.0, 0.0, 0.0, -1.5, 0.0]) + 0.3)


def test_a_tiny_price_recovers_the_weights() -> None:
    columns, targets = synthetic()

    fit = SparseFitter().fit(columns, targets, price=1e-6, max_steps=5000, tolerance=1e-8)

    assert fit.settled
    assert fit.weights == pytest.approx((2.0, 0.0, 0.0, -1.5, 0.0), abs=0.01)
    assert fit.bias == pytest.approx(0.3, abs=0.01)


def test_a_price_keeps_the_terms_with_an_effect_shrunk_and_drops_the_others() -> None:
    columns, targets = synthetic()

    fit = SparseFitter().fit(columns, targets, price=0.01, max_steps=5000, tolerance=1e-8)

    assert fit.settled
    assert [index for index, weight in enumerate(fit.weights) if weight != 0.0] == [0, 3]
    assert 0.0 < fit.weights[0] < 2.0 and -1.5 < fit.weights[3] < 0.0


def test_a_column_s_cost_multiplies_its_price() -> None:
    columns, targets = synthetic()

    fit = SparseFitter().fit(columns, targets, 0.01, 5000, 1e-8, costs=np.array([1.0, 1.0, 1.0, 1000.0, 1.0]))

    assert [index for index, weight in enumerate(fit.weights) if weight != 0.0] == [0]


def test_a_high_price_drops_every_term_and_leaves_the_bias_to_the_mean_target() -> None:
    columns, targets = synthetic()

    fit = SparseFitter().fit(columns, targets, price=10.0, max_steps=5000, tolerance=1e-10)

    mean = float(targets.mean())
    assert fit.weights == (0.0,) * 5
    assert fit.bias == pytest.approx(math.log(mean / (1.0 - mean)), abs=1e-4)


def test_a_fit_stops_at_the_step_limit_and_can_resume_from_there() -> None:
    columns, targets = synthetic()
    fitter = SparseFitter()

    first = fitter.fit(columns, targets, 0.01, max_steps=3, tolerance=1e-12)
    resumed = fitter.fit(columns, targets, 0.01, max_steps=5000, tolerance=1e-8, start=first)

    assert (first.steps, first.settled) == (3, False)
    assert resumed.settled
    assert fitter.loss(columns, targets, resumed.weights, resumed.bias) < fitter.loss(columns, targets, first.weights, first.bias)


def test_loss_is_the_mean_logistic_loss_without_the_price() -> None:
    columns, targets = np.array([[0.0], [1.0]]), np.array([0.5, 1.0])

    loss = SparseFitter().loss(columns, targets, (2.0,), 0.0)

    assert loss == pytest.approx((math.log(2.0) + math.log(1.0 + math.exp(2.0)) - 2.0) / 2)


def test_a_fit_needs_rows() -> None:
    with pytest.raises(ValueError, match="row"):
        SparseFitter().fit(np.empty((0, 2)), np.empty(0), 0.1, 10, 1e-6)
