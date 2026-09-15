"""Folding an aggregate from its readings must give exactly what running its source gives.

The search may only share readings between candidates if the shared result is the same, so these tests take real
generated aggregates on three games and compare both ways of reading them, column by column.
"""

import numpy as np
import pytest

from openmind.agent.model.domain import Domain
from openmind.inference.model.expression import Expression
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_generator_tests import line_domain, line_position
from openmind.inference.service.position_view_tests import capture_domain, capture_position
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.consequence_library_tests import position, strip_domain
from openmind.rbs.service.term_evaluator import AggregateParts, TermEvaluator
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

pytestmark = pytest.mark.log_level("INFO")


def parts(expression: Expression) -> AggregateParts:
    aggregate = expression.aggregate
    assert aggregate is not None
    return (aggregate.base, aggregate.kind, aggregate.pair, aggregate.readings, aggregate.operations)


def aggregates(domain: Domain, rows: tuple[PositionRow, ...], generations: int = 2) -> list[Expression]:
    """Every aggregate a few generations of growth give, the leaves first."""
    generator = ExpressionGenerator(VariableNameMapper())
    vocabulary = generator.vocabulary(domain, (row.state for row in rows))
    found = [leaf for leaf in generator.leaves(vocabulary) if leaf.aggregate is not None]
    grown = list(found)
    for _ in range(generations - 1):
        children = [child for parent in grown for child in generator.aggregate_children(parent, vocabulary)]
        grown = children[:60]
        found.extend(grown)
    return found


def check(domain: Domain, rows: tuple[PositionRow, ...], evaluator: TermEvaluator) -> tuple[int, int]:
    """How many aggregates were folded, and how many the fold couldn't take, having compared every folded one with its
    source's column."""
    generator = ExpressionGenerator(VariableNameMapper())
    found = aggregates(domain, rows)
    folded = evaluator.aggregate_columns(domain, rows, [parts(expression) for expression in found])
    ran = evaluator.columns(domain, rows, [generator.source(expression) for expression in found])
    left_out = 0
    for expression, one, other in zip(found, folded, ran, strict=True):
        if one is None:
            left_out += 1
            continue
        assert other is not None, f"the source gives no column for {expression.template}"
        assert np.allclose(one, other, equal_nan=True), f"folding differs from the source for {expression.template}"
    return len(found) - left_out, left_out


def test_folding_reads_the_strip_as_its_source_does() -> None:
    rows = tuple(
        PositionRow(state, player, 0.0)
        for state in (position({1: "X", 2: "O"}, "X"), position({1: "X"}, "O"), position({}, "X"))
        for player in ("X", "O")
    )

    folded, left_out = check(strip_domain(), rows, new_evaluator())

    assert folded > 20 and left_out >= 0


def test_folding_reads_a_game_without_a_board_as_its_source_does() -> None:
    rows = tuple(
        PositionRow(state, player, 0.0)
        for state in (line_position(0.0, 2.0, "A"), line_position(1.25, -0.5, "B"))
        for player in ("A", "B")
    )

    folded, _ = check(line_domain(), rows, new_evaluator())

    assert folded > 0


def test_folding_reads_the_capture_game_as_its_source_does() -> None:
    rows = tuple(
        PositionRow(state, player, 0.0)
        for state in (capture_position({2: "A", 6: "A", 4: "B", 7: "B"}, "A"), capture_domain().initial_state)
        for player in ("A", "B")
    )

    folded, _ = check(capture_domain(), rows, new_evaluator())

    assert folded > 0


def test_an_aggregate_over_pairs_or_without_parts_is_left_to_its_source() -> None:
    evaluator, rows = new_evaluator(), (PositionRow(position({1: "X"}, "O"), "X", 0.0),)

    (pairs,) = evaluator.aggregate_columns(strip_domain(), rows, [("cell", "sum", True, ("{view}.cell[i] == me",), ())])
    (partless,) = evaluator.aggregate_columns(strip_domain(), rows, [("cell", "sum", False, (), ())])

    assert pairs is None and partless is None
