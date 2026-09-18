import numpy as np
import pytest

from openmind.rbs.service.sparse_fitter import SparseFitter


def synthetic() -> tuple[np.ndarray, np.ndarray]:
    """Five standard normal columns; the payoffs are a linear reading of the first and the fourth, plus a little
    noise, which is the shape a position heuristic is fitted to."""
    rng = np.random.default_rng(1)
    columns = rng.normal(size=(2000, 5))
    return columns, columns @ np.array([2.0, 0.0, 0.0, -1.5, 0.0]) + 0.3 + rng.normal(scale=0.01, size=2000)


def test_a_tiny_price_recovers_the_weights_in_the_payoffs_own_units() -> None:
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
    dear = np.array([1.0, 1.0, 1.0, 1000.0, 1.0])

    fit = SparseFitter().fit(columns, targets, 0.01, 5000, 1e-8, costs=dear)
    same = SparseFitter().fit(columns, targets, 0.01, 5000, 1e-8)

    # The fourth term reads the payoffs as surely as the first, and is kept at the same price; priced a thousand
    # times over, it doesn't pay for itself and drops out.
    assert same.weights[3] != 0.0
    assert fit.weights[3] == 0.0
    assert fit.weights[0] == pytest.approx(2.0, abs=0.05)


def test_a_high_price_drops_every_term_and_leaves_the_bias_at_the_mean_payoff() -> None:
    columns, targets = synthetic()

    fit = SparseFitter().fit(columns, targets, price=10.0, max_steps=5000, tolerance=1e-10)

    assert fit.weights == (0.0,) * 5
    assert fit.bias == pytest.approx(float(targets.mean()), abs=1e-4)


def test_a_fit_stops_at_the_step_limit_and_can_resume_from_there() -> None:
    columns, targets = synthetic()
    fitter = SparseFitter()

    first = fitter.fit(columns, targets, 0.01, max_steps=3, tolerance=1e-12)
    resumed = fitter.fit(columns, targets, 0.01, max_steps=5000, tolerance=1e-8, start=first)

    assert (first.steps, first.settled) == (3, False)
    assert resumed.settled
    assert fitter.loss(columns, targets, resumed.weights, resumed.bias) < fitter.loss(columns, targets, first.weights, first.bias)


def test_loss_is_the_mean_squared_error_without_the_price() -> None:
    columns, targets = np.array([[0.0], [1.0]]), np.array([0.5, 1.0])

    loss = SparseFitter().loss(columns, targets, (2.0,), 0.0)

    assert loss == pytest.approx((0.5**2 + 1.0**2) / 2)


def test_a_fit_needs_rows() -> None:
    with pytest.raises(ValueError, match="row"):
        SparseFitter().fit(np.empty((0, 2)), np.empty(0), 0.1, 10, 1e-6)
