import math
from collections.abc import Callable

import numpy as np
import pytest

from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner

type Game = Callable[[str], RuleBasedGame]


def evaluating(workers: int = 1) -> TermEvaluator:
    """A term evaluator wired as the heuristic finder's builder wires one, in this process."""
    compiler, runner = RuleCompiler(), RuleRunner(StateNamespaceMapper())
    library = ConsequenceLibraryBuilder().build()
    return TermEvaluator(compiler, runner, library, TaskRunner(workers), ReadingCache(compiler, runner, library, MemoryMeter()))


def rows(played: RuleBasedGame) -> list[PositionRow]:
    start = played.start()
    reached = [start, *(played.outcomes(start, action).outcomes[0][0] for action in played.actions(start)[:3])]
    return [PositionRow(state, "X", 0.0) for state in reached]


def test_a_term_is_read_on_every_row_in_the_rows_order(game: Game) -> None:
    played = game("tictactoe")

    column = evaluating().column(played, rows(played), PythonRule("1.0"))

    assert list(column) == [1.0, 1.0, 1.0, 1.0]


def test_a_boolean_counts_as_one_or_nothing():
    """Most of what a heuristic reads is a condition, and a fit needs a number: true is one of the thing, false
    is none of it."""
    evaluator = evaluating()

    assert evaluator.number(True) == 1.0
    assert evaluator.number(False) == 0.0


def test_a_term_reading_nothing_here_leaves_a_blank_rather_than_a_nought(game: Game) -> None:
    """A fork detector in a position with no fork has not found zero forks worth of something; it has nothing
    to say. A nought would be averaged in as an opinion nobody held."""
    played = game("tictactoe")

    column = evaluating().column(played, rows(played), PythonRule("None"))

    assert all(math.isnan(one) for one in column)


def test_a_term_that_raises_anywhere_is_dropped_rather_than_read_as_nothing(game: Game) -> None:
    """A term reading a model the game has not is not a term that reads nothing — it is a term that does not
    belong to this game, and keeping it at nought would let it into a fit."""
    played = game("tictactoe")

    assert evaluating().column(played, rows(played), PythonRule("lamp")) is None
    assert evaluating().column(played, rows(played), PythonRule("1 / 0")) is None


def test_a_term_giving_something_that_is_not_a_number_is_dropped(game: Game) -> None:
    played = game("tictactoe")

    assert evaluating().column(played, rows(played), PythonRule("'a mark'")) is None


def test_a_number_that_is_not_finite_is_not_a_number_a_fit_can_use():
    """An infinity in a column makes every weight in the fit infinite, and a NaN makes them all NaN — one bad
    row would take the whole heuristic with it."""
    evaluator = evaluating()

    assert evaluator.number(float("inf")) is None
    assert evaluator.number(float("nan")) is None
    assert evaluator.number(2.5) == 2.5


def test_several_terms_come_back_in_the_order_they_were_asked_for(game: Game) -> None:
    """The columns are matched to their terms by position, so a reordering would silently give every term
    another's weight."""
    played = game("tictactoe")
    asked = [PythonRule("1.0"), PythonRule("2.0"), PythonRule("3.0")]

    found = evaluating().columns(played, rows(played), asked)

    assert [one[0] for one in found] == [1.0, 2.0, 3.0]


def test_a_term_that_fails_leaves_its_own_place_empty_and_not_the_others(game: Game) -> None:
    played = game("tictactoe")

    found = evaluating().columns(played, rows(played), [PythonRule("1.0"), PythonRule("lamp"), PythonRule("3.0")])

    assert found[1] is None
    assert found[0] is not None and found[2] is not None


def test_asking_for_no_terms_gives_no_columns(game: Game) -> None:
    played = game("tictactoe")

    assert evaluating().columns(played, rows(played), []) == []


def test_what_a_term_took_to_read_is_measured_as_it_is_read(game: Game) -> None:
    """A term is not only worth what it explains, it costs what it takes to read — and nothing was measuring
    that, so a look-ahead reading the position after every legal action was priced as one more clause."""
    played = game("tictactoe")
    evaluator = evaluating()
    term = PythonRule("1.0")

    assert evaluator.seconds(term) == 0.0
    evaluator.column(played, rows(played), term)

    assert evaluator.seconds(term) > 0.0


def test_a_dear_term_is_dearer_than_a_cheap_one_by_what_it_actually_took(game: Game) -> None:
    """Nothing here knows what a look-ahead is or that chess has thirty-five moves. It knows that this term
    took longer than that one on the same rows, which is true of whatever made it slow."""
    played = game("tictactoe")
    evaluator = evaluating()
    cheap, dear = PythonRule("1.0"), PythonRule("sum(i * i for i in range(20000))")

    for term in (cheap, dear):
        evaluator.column(played, rows(played), term)

    assert evaluator.dearness(cheap) == pytest.approx(1.0)
    assert evaluator.dearness(dear) > 10.0


def test_a_term_nobody_timed_is_not_taken_to_be_dear(game: Game) -> None:
    """A term folded from readings rather than run never passes through here, and charging it for a
    measurement that was never taken would price it out for not having been measured."""
    assert evaluating().dearness(PythonRule("never read")) == 1.0


def test_dearness_is_relative_so_it_says_the_same_on_a_slow_machine(game: Game) -> None:
    """Seconds on a fast machine and seconds on a slow one are different numbers about the same term. What a
    fit needs is how a term compares with the others it is competing with."""
    played = game("tictactoe")
    evaluator = evaluating()
    for term in (PythonRule("1.0"), PythonRule("2.0")):
        evaluator.column(played, rows(played), term)

    cheapest = min(evaluator.seconds(PythonRule(one)) for one in ("1.0", "2.0"))

    assert cheapest > 0.0
    assert min(evaluator.dearness(PythonRule(one)) for one in ("1.0", "2.0")) == pytest.approx(1.0)
