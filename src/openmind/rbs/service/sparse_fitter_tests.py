import numpy as np
import pytest

from openmind.rbs.model.sparse_fit import SparseFit
from openmind.rbs.service.sparse_fitter import SparseFitter

#: Enough steps and a tight enough tolerance that a fit settles on its own rather than running out.
STEPS, TOLERANCE = 5000, 1e-9


def a_line(rows: int = 40, slope: float = 3.0, bias: float = 1.0):
    """One column that predicts the target exactly, so what is fitted is known before fitting it."""
    columns = np.linspace(-1.0, 1.0, rows).reshape(rows, 1)
    return columns, (columns[:, 0] * slope + bias)


def test_a_term_that_predicts_the_target_is_fitted_to_what_it_predicts_by():
    """Nothing priced, nothing to trade off: the weight is the slope and the bias is the intercept."""
    columns, targets = a_line()

    fitted = SparseFitter().fit(columns, targets, 0.0, STEPS, TOLERANCE)

    assert fitted.weights[0] == pytest.approx(3.0, abs=1e-3)
    assert fitted.bias == pytest.approx(1.0, abs=1e-3)
    assert fitted.settled


def test_a_term_that_does_not_pay_its_price_is_dropped_to_exactly_nothing():
    """A weight of exactly 0 drops its term, which is what the price is for: shrinking toward 0 and landing on
    it, rather than leaving a term at some small weight that still has to be read every time."""
    columns, targets = a_line(slope=0.02)

    fitted = SparseFitter().fit(columns, targets, 1.0, STEPS, TOLERANCE)

    assert fitted.weights[0] == 0.0


def test_the_bias_is_never_priced():
    """What a position is worth before anything is read is not a term anybody pays for, so a game whose payoffs
    are all the same still says so instead of being shrunk to zero."""
    rows = 20
    columns = np.zeros((rows, 1))
    targets = np.full(rows, 0.75)

    fitted = SparseFitter().fit(columns, targets, 10.0, STEPS, TOLERANCE)

    assert fitted.bias == pytest.approx(0.75, abs=1e-3)


def test_a_dearer_price_keeps_fewer_terms():
    """The sweep the value generator runs rests on this being monotone: raise the price and terms drop out."""
    rows = 60
    generator = np.random.default_rng(7)
    columns = generator.normal(size=(rows, 6))
    targets = columns[:, 0] * 2.0 + columns[:, 1] * 0.4 + generator.normal(scale=0.05, size=rows)
    fitter = SparseFitter()

    kept = [
        sum(1 for weight in fitter.fit(columns, targets, price, STEPS, TOLERANCE).weights if weight != 0.0)
        for price in (0.0, 0.2, 1.0, 5.0)
    ]

    assert kept == sorted(kept, reverse=True)
    assert kept[0] > kept[-1]


def test_a_term_costing_more_is_priced_more():
    """A cost per column is how a term that reads a whole board is made to pay for reading it, so two terms
    that predict alike are told apart by what they cost rather than by which came first."""
    rows = 80
    generator = np.random.default_rng(3)
    first = generator.normal(size=rows)
    columns = np.column_stack([first, first + generator.normal(scale=1e-6, size=rows)])
    targets = first * 1.0

    fitted = SparseFitter().fit(columns, targets, 0.05, STEPS, TOLERANCE, costs=np.array([1.0, 50.0]))

    assert abs(fitted.weights[0]) > abs(fitted.weights[1])


def test_a_fit_may_be_started_from_weights_somebody_else_reached():
    """What a seeded search hands the fitter: the weight the rules imply a term should start at. It buys the
    first step, not the answer — where the gradient disagrees the fit still moves away from it."""
    columns, targets = a_line()
    started = SparseFit((3.0,), 1.0, 0, True)

    fitted = SparseFitter().fit(columns, targets, 0.0, STEPS, TOLERANCE, start=started)

    assert fitted.weights[0] == pytest.approx(3.0, abs=1e-3)
    assert fitted.steps < SparseFitter().fit(columns, targets, 0.0, STEPS, TOLERANCE).steps


def test_a_seed_the_evidence_disagrees_with_is_moved_off_regardless_of_where_it_started():
    """The whole honesty of seeding: a weight starting high is not a weight kept high."""
    columns, targets = a_line(slope=0.0, bias=0.0)

    fitted = SparseFitter().fit(columns, targets, 0.5, STEPS, TOLERANCE, start=SparseFit((9.0,), 0.0, 0, True))

    assert fitted.weights[0] == 0.0


def test_a_fit_that_runs_out_of_steps_says_it_did_not_settle():
    """Thin evidence looking like a settled answer is what sends somebody trusting a number that was still
    moving when the clock stopped."""
    columns, targets = a_line()

    fitted = SparseFitter().fit(columns, targets, 0.0, 2, 1e-12)

    assert fitted.steps == 2
    assert not fitted.settled


def test_a_fit_with_no_rows_says_so_rather_than_giving_an_answer():
    with pytest.raises(ValueError, match="at least one row"):
        SparseFitter().fit(np.empty((0, 2)), np.empty(0), 0.0, STEPS, TOLERANCE)


def test_the_loss_is_what_the_weights_leave_and_carries_no_price():
    """The sweep chooses on held-out loss, so the loss must be comparable between prices — which it only is
    where the price is left out of it."""
    columns, targets = a_line()
    fitter = SparseFitter()

    assert fitter.loss(columns, targets, (3.0,), 1.0) == pytest.approx(0.0, abs=1e-9)
    assert fitter.loss(columns, targets, (0.0,), 0.0) == pytest.approx(float(np.mean(targets**2)))
