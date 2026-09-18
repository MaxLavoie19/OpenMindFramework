import itertools
import logging
from collections.abc import Callable, Iterator
from dataclasses import replace

import numpy as np
import pytest

from openmind.inference.model.expression import Expression
from openmind.inference.model.search_budget import SearchBudget
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_generator_tests import line_domain, line_position
from openmind.inference.service.expression_search import ExpressionSearch
from openmind.parallel.model.call_over_memory import CallOverMemory
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.consequence_library_tests import Declare, strip_domain
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner
from openmind.rbs.service.sparse_fitter import SparseFitter
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.rbs.service.value_generator_tests import strip_rows
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players

pytestmark = pytest.mark.log_level("INFO")

GIGABYTE = 1024**3


class CountingEvaluator(TermEvaluator):
    """A term evaluator counting how often it is asked to clear its memory."""

    def __init__(self) -> None:
        super().__init__(
            RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ConsequenceLibraryBuilder().build(), TaskRunner(1)
        )
        self.clears = 0

    def clear_memory(self) -> None:
        self.clears += 1
        super().clear_memory()


class RecordingEvaluator(TermEvaluator):
    """A term evaluator recording the sources of every batch it evaluates."""

    def __init__(self) -> None:
        super().__init__(
            RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ConsequenceLibraryBuilder().build(), TaskRunner(1)
        )
        self.batches: list[list[PythonRule]] = []

    def columns(self, rbs, rows, sources):  # type: ignore[no-untyped-def]
        self.batches.append(list(sources))
        return super().columns(rbs, rows, sources)


class OverMemoryEvaluator(TermEvaluator):
    """A term evaluator whose workers go over their memory cap twice on every batch."""

    def __init__(self) -> None:
        super().__init__(
            RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ConsequenceLibraryBuilder().build(), TaskRunner(1)
        )

    def columns(self, rbs, rows, sources):  # type: ignore[no-untyped-def]
        raise CallOverMemory(0, None)


class EndlessGenerator(ExpressionGenerator):
    """An expression generator whose look-aheads never end."""

    def look_aheads(self, expression: Expression) -> Iterator[Expression]:  # type: ignore[override]
        return (
            Expression(f"({expression.template}) + {step}", expression.clauses + 1, expression.plies)
            for step in itertools.count(1)
        )


def new_search(
    clock: Callable[[], float] | None = None,
    evaluator: TermEvaluator | None = None,
    generator: ExpressionGenerator | None = None,
) -> ExpressionSearch:
    parts = (
        generator or ExpressionGenerator(VariableNameMapper()),
        evaluator or new_evaluator(),
        SparseFitter(),
        MemoryMeter(),
    )
    return ExpressionSearch(*parts) if clock is None else ExpressionSearch(*parts, clock)


def targets() -> np.ndarray:
    return np.array([row.target for row in strip_rows()])


def test_kept_expressions_look_ahead_and_their_columns_are_their_sources_values(declared: Declare, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.inference")
    rows, rbs, search = strip_rows(), strip_domain(declared), new_search()

    result = search.search(rbs, rows, rows[:5], targets(), 0.01, 500, 1e-6, SearchBudget(300.0, GIGABYTE, 3000))

    assert result.expressions and max(expression.plies for expression in result.expressions) >= 1
    generator = ExpressionGenerator(VariableNameMapper())
    sources = [generator.source(expression) for expression in result.expressions]
    for column, held, evaluated, evaluated_held in zip(
        result.training,
        result.held_out,
        new_evaluator().columns(rbs, rows, sources),
        new_evaluator().columns(rbs, rows[:5], sources),
        strict=True,
    ):
        assert evaluated is not None and evaluated_held is not None
        assert np.array_equal(column, evaluated) and np.array_equal(held, evaluated_held)
    assert any(message.startswith("Generation 2: ") for message in caplog.messages)
    assert any(message.startswith("Search stopped after ") for message in caplog.messages)


def test_without_a_board_the_search_relates_real_positions(declared: Declare) -> None:
    """On a line where nobody can move, no look-ahead can stand in for the distance between the tokens, so the kept
    expressions must read both tokens together: a gap, a difference, or aggregates over every token, which the fit can
    weigh into one (the highest token at about -3.5 and the lowest at about +3.4 is minus the gap)."""
    rng = np.random.default_rng(3)
    positions = rng.uniform(-3.0, 3.0, size=(400, 2))
    rows = [
        PositionRow(line_position(float(a), float(b), "A"), "A", 1.0 if abs(a - b) < 1.0 else 0.0) for a, b in positions
    ]
    scaled = np.array([row.target for row in rows])
    frozen = declared(
        line_position(0.0, 2.0, "A"),
        players=Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
        context="line without moves",
    )

    result = new_search().search(frozen, rows, (), scaled, 0.01, 1000, 1e-6, SearchBudget(300.0, GIGABYTE, 5000))

    def relates_both(template: str) -> bool:
        every_token = "for i in {view}.x" in template
        first = "x['A']" in template or "x[me]" in template
        second = "x['B']" in template or "x[other]" in template
        return every_token or (first and second)

    templates = [expression.template for expression in result.expressions]
    assert any(relates_both(template) for template in templates)
    assert not any("offset(" in template for template in templates)


def test_the_search_stops_when_it_has_tried_its_candidates_even_when_they_never_end(declared: Declare) -> None:
    result = new_search(generator=EndlessGenerator(VariableNameMapper())).search(
        strip_domain(declared), strip_rows(), (), targets(), 0.01, 500, 1e-6, SearchBudget(300.0, GIGABYTE, 600)
    )

    assert (result.stopped, result.tried) == ("the candidate budget ran out", 600)


def test_seeds_are_tried_before_the_leaves(declared: Declare) -> None:
    evaluator = RecordingEvaluator()
    seed = Expression("{view}.mobility(other)", 1, 0)

    new_search(evaluator=evaluator).search(
        strip_domain(declared), strip_rows(), (), targets(), 0.01, 500, 1e-6, SearchBudget(300.0, GIGABYTE, 200), (seed,)
    )

    assert evaluator.batches[0][0] == PythonRule("here.mobility(other)")


def test_a_generation_that_keeps_nothing_leaves_the_next_to_build_on_what_it_passed_over(declared: Declare) -> None:
    result = new_search().search(strip_domain(declared), strip_rows(), (), targets(), 1e6, 500, 1e-6, SearchBudget(300.0, GIGABYTE, 3000))

    assert result.expressions == ()
    assert result.generations >= 2
    assert result.stopped in ("nothing left to try", "the candidate budget ran out")


def test_the_search_stops_when_its_time_runs_out(declared: Declare) -> None:
    times = itertools.chain((0.0, 0.0), itertools.repeat(5.0))

    result = new_search(lambda: next(times)).search(
        strip_domain(declared), strip_rows(), (), targets(), 0.01, 500, 1e-6, SearchBudget(1.0, GIGABYTE)
    )

    assert (result.generations, result.stopped) == (1, "the time budget ran out")
    assert result.expressions and all(expression.plies == 0 for expression in result.expressions)


def test_a_search_whose_process_holds_more_than_its_memory_budget_clears_its_views_and_stops(declared: Declare, 
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.inference")
    evaluator = CountingEvaluator()

    result = new_search(evaluator=evaluator).search(
        strip_domain(declared), strip_rows(), (), targets(), 0.01, 500, 1e-6, SearchBudget(300.0, 1, 3000)
    )

    assert (result.stopped, result.tried, result.expressions, evaluator.clears) == ("the memory budget ran out", 0, (), 1)
    assert any(", over the memory budget of 1: cleared the views" in message for message in caplog.messages)


def first_generation_budget(declared: Declare) -> SearchBudget:
    """A budget of exactly the first generation's candidates, the strip's leaves."""
    generator = ExpressionGenerator(VariableNameMapper())
    leaves = generator.leaves(generator.vocabulary(strip_domain(declared), (row.state for row in strip_rows())))
    return SearchBudget(300.0, GIGABYTE, len(leaves))


def test_a_candidate_is_kept_when_any_target_supports_it(declared: Declare, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.inference")
    noise = np.random.default_rng(1).uniform(0.0, 1.0, len(strip_rows()))

    alone = new_search().search(strip_domain(declared), strip_rows(), (), targets(), 0.01, 500, 1e-6, first_generation_budget(declared))
    both = new_search().search(
        strip_domain(declared), strip_rows(), (), {"payoff": targets(), "noise": noise}, 0.01, 500, 1e-6, first_generation_budget(declared)
    )

    alone_templates = {expression.template for expression in alone.expressions}
    assert alone_templates and alone_templates <= {expression.template for expression in both.expressions}
    assert any("training loss payoff=" in message and " noise=" in message for message in caplog.messages)


def test_one_named_target_searches_as_the_target_given_alone(declared: Declare) -> None:
    named = new_search().search(strip_domain(declared), strip_rows(), (), {"payoff": targets()}, 0.01, 500, 1e-6, first_generation_budget(declared))
    alone = new_search().search(strip_domain(declared), strip_rows(), (), targets(), 0.01, 500, 1e-6, first_generation_budget(declared))

    assert (named.expressions, named.stopped, named.tried) == (alone.expressions, alone.stopped, alone.tried)


def test_a_search_without_a_target_raises(declared: Declare) -> None:
    with pytest.raises(ValueError, match="at least one target"):
        new_search().search(strip_domain(declared), strip_rows(), (), {}, 0.01, 500, 1e-6, first_generation_budget(declared))


def test_a_column_with_blanks_is_scaled_without_centering_and_priced_by_its_share(declared: Declare) -> None:
    search = new_search()
    fires = np.array([1.0, np.nan, 1.0, np.nan])
    counts = np.array([1.0, 3.0, 1.0, 3.0])

    assert search.scaling(fires) == (0.0, 1.0)
    assert search.standard(fires).tolist() == [1.0, 0.0, 1.0, 0.0]
    assert search.share(fires) == 0.5
    assert search.standard(counts).tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert search.share(counts) == 1.0
    assert search.standard(np.array([np.nan, np.nan])).tolist() == [0.0, 0.0]


def test_a_blank_term_is_kept_and_what_combines_it_here_stays_blank_where_it_is(declared: Declare) -> None:
    seed = Expression("(1 if wins(me) else None)", 1, 0)

    result = new_search().search(
        strip_domain(declared), strip_rows(), strip_rows()[:5], targets(), 0.01, 500, 1e-6, SearchBudget(300.0, GIGABYTE, 3000), (seed,)
    )

    columns = dict(zip((expression.template for expression in result.expressions), result.training, strict=True))
    assert seed.template in columns
    blank = np.isnan(columns[seed.template])
    assert blank.any() and not blank.all()
    # A child combining the term's reading in this position is blank wherever the term is. A look-ahead is not: it
    # reads the term after a move, and a position the term says nothing about can lead to ones it does.
    combined = [
        column
        for template, column in columns.items()
        if template != seed.template and seed.template in template and "lambda" not in template
    ]
    ahead = [
        column
        for template, column in columns.items()
        if seed.template in template and "lambda" in template
    ]
    assert combined and all(np.isnan(column[blank]).all() for column in combined)
    assert ahead and any(not np.isnan(column[blank]).all() for column in ahead)


def test_a_search_whose_workers_go_over_their_memory_cap_twice_stops_on_memory(declared: Declare, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="openmind.inference")

    result = new_search(evaluator=OverMemoryEvaluator()).search(
        strip_domain(declared), strip_rows(), (), targets(), 0.01, 500, 1e-6, SearchBudget(300.0, 64 * GIGABYTE, 3000)
    )

    assert (result.stopped, result.expressions) == ("the memory budget ran out", ())
    assert any(message.startswith("Evaluating candidates took a worker over its memory cap twice") for message in caplog.messages)
