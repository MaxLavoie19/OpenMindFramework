import numpy as np
import pytest

from openmind.inference.constant.inference_constant import SEEDED_WEIGHT
from openmind.inference.model.expression import Expression
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_search import ExpressionSearch
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.model.sparse_fit import SparseFit
from openmind.rbs.service.sparse_fitter import SparseFitter


def a_search():
    return ExpressionSearch(ExpressionGenerator(), None, SparseFitter(), MemoryMeter())  # type: ignore[arg-type]


def _timed(search, dearness: float):
    """A search whose terms all take that long to read, since `cost` asks what a term was timed at."""
    search._term_evaluator = _Timing(dearness)  # noqa: SLF001
    return search


class _Timing:
    def __init__(self, dearness: float) -> None:
        self._dearness = dearness

    def dearness(self, source: str) -> float:
        return self._dearness


def a_term(template):
    return Expression(template, 1, 0)


def columns(values):
    held = np.asarray(values, dtype=float)
    return (held, held)


def test_a_seeded_term_starts_the_fit_from_what_the_rules_said_it_is_worth():
    """A linear heuristic values a position as the sum of its weighted readings, so the weight on a term *is*
    what the thing it counts is worth. The reasoned number is not carried alongside the term — it starts it."""
    search = a_search()
    kept = [columns([0.0, 1.0, 2.0, 3.0])]

    started = search._started([a_term("knights")], kept, {"knights": 1.0})

    assert started is not None
    assert started.weights[0] != 0.0


def test_a_kept_term_nothing_seeded_starts_where_everything_else_does():
    """Only the seeds carry a weight in; a term the search found for itself begins at nothing, as it did."""
    search = a_search()
    kept = [columns([0.0, 1.0, 2.0, 3.0]), columns([3.0, 1.0, 4.0, 1.0])]

    started = search._started([a_term("knights"), a_term("something else")], kept, {"knights": 1.0})

    assert started is not None
    assert started.weights[1] == 0.0


def test_nothing_is_started_where_no_kept_term_was_seeded():
    """A search with no seeds fits exactly as it did before there were any."""
    search = a_search()

    assert search._started([a_term("knights")], [columns([0.0, 1.0])], {}) is None
    assert search._started([a_term("knights")], [columns([0.0, 1.0])], {"other": 4.0}) is None


def test_a_reasoned_count_is_put_on_the_scale_the_fit_works_on_and_cannot_swamp_it():
    """A rook reaching fourteen squares is a count of squares. The fit works on centered, scaled columns with
    its prediction pushed through a logistic, where fourteen means nothing — so it is scaled by the column's
    own scale and held to where a weight on that scale sensibly lives. A single reasoned number must not be
    able to swamp the fit it is only meant to start."""
    search = a_search()

    started = search._started([a_term("rooks")], [columns([0.0, 10.0, 20.0, 30.0])], {"rooks": 14.0})

    assert started is not None
    assert abs(started.weights[0]) == SEEDED_WEIGHT


def test_a_term_that_never_varies_has_nothing_for_a_weight_to_start_on():
    """A column that reads the same everywhere is scaled to all zeros and tells no two rows apart, so a weight
    on it would be a number sitting on nothing."""
    search = a_search()

    assert search._started([a_term("always four")], [columns([4.0, 4.0, 4.0])], {"always four": 9.0}) is None


def test_a_seeded_weight_does_not_buy_a_term_past_the_price():
    """The whole filter depends on this. A term whose gradient does not pay is shrunk to nothing whatever
    weight it started at — so seeding cannot smuggle a worthless term into a heuristic, and what a seed buys
    is being expanded first rather than being kept."""
    drawn = np.random.default_rng(7)
    noise = drawn.normal(size=200)
    targets = (drawn.normal(size=200) > 0).astype(float)
    standard = np.column_stack([(noise - noise.mean()) / noise.std()])

    started = SparseFit((SEEDED_WEIGHT,), 0.0, 0, False)
    fitted = SparseFitter().fit(standard, targets, 0.1, 200, 1e-4, started, np.array([1.0]))

    assert fitted.weights[0] == 0.0


def test_a_term_that_reads_nought_has_spoken_but_has_not_fired() -> None:
    """The distinction the whole detector problem turns on. A blank is a term that was not read; a nought is a
    term that looked and found nothing. A mate detector reads nought on nearly every position, so counted by
    `share` it spoke everywhere and counted by `firing` it spoke twice."""
    search = a_search()
    column = np.array([0.0, 0.0, 1.0, 0.0, np.nan, 1.0, 0.0, 0.0, 0.0, 0.0])

    assert search.share(column) == pytest.approx(0.9), "nine of ten rows were read"
    assert search.firing(column) == pytest.approx(0.2), "and on two of them it had something to say"


def test_a_rare_decisive_term_is_not_priced_as_though_it_spoke_everywhere() -> None:
    """Charged on `share` alone, a term that fires on a fiftieth of rows pays as much per unit of weight as
    one that fires on half of them, while paying off on a fiftieth as many rows — so the price takes it out
    first. Measured at price 0.05 with the decisive strength held at 0.5, a term firing on 1% of rows came out
    at 0.011 charged on share and 0.493 charged on both.

    This is the adaptive lasso, and the point of multiplying rather than replacing is below."""
    search = _timed(a_search(), 1.0)
    rare = np.where(np.arange(100) < 2, 1.0, 0.0)
    common = np.where(np.arange(100) < 50, 1.0, 0.0)
    term = Expression(template="VIEW", clauses=2, plies=0)

    assert search.cost(term, rare) < search.cost(term, common)


def test_what_a_term_costs_to_read_is_still_charged_when_it_fires_rarely() -> None:
    """Why firing multiplies `share` instead of replacing it. A look-ahead reads the position after every
    legal action on every position, whether or not it finds anything there, and the plain adaptive lasso would
    charge it only for the rows it fired on — forgiving a cost that is really paid. Two terms that fire
    equally rarely and differ only in what they cost to read must not price alike."""
    rare = np.where(np.arange(100) < 2, 1.0, 0.0)
    term = Expression(template="VIEW", clauses=2, plies=0)

    cheap = _timed(a_search(), 1.0).cost(term, rare)
    dear = _timed(a_search(), 35.0).cost(term, rare)

    assert dear > cheap, "a look-ahead reads every position whether it finds anything there or not"
