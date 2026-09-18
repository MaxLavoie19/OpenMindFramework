import itertools
import logging
import math
from dataclasses import replace

import numpy as np
import pytest

from openmind.inference.model.expression import Expression
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system, create_value_generator
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.service.value_generator import CONSTANT_RULE
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.consequence_library_tests import Declare, position, strip_domain

pytestmark = pytest.mark.log_level("INFO")

SETTINGS = ValueSettings(
    prices=(0.1, 0.01, 0.001), max_steps=2000, tolerance=1e-6, seconds=300.0, memory_bytes=1024**3, candidates=3000
)


def strip_rows() -> tuple[PositionRow, ...]:
    """Every strip position with an empty cell and no win yet, for each player: when the player to act can win now,
    1.0 for them and 0.0 for the other; otherwise 0.5 for both."""
    rows: list[PositionRow] = []
    for marks in itertools.product((None, "X", "O"), repeat=4):
        crosses, noughts = marks.count("X"), marks.count("O")
        won = any(mark is not None and mark == following for mark, following in itertools.pairwise(marks))
        if crosses - noughts not in (0, 1) or won or None not in marks:
            continue
        turn = "X" if crosses == noughts else "O"
        can_win = any(
            marks[col] is None and turn in (marks[col - 1] if col > 0 else None, marks[col + 1] if col < 3 else None)
            for col in range(4)
        )
        state = position({col + 1: mark for col, mark in enumerate(marks) if mark is not None}, turn)
        for player in ("X", "O"):
            rows.append(PositionRow(state, player, (1.0 if player == turn else 0.0) if can_win else 0.5))
    return tuple(rows)


def mean_absolute_error(values: list[float], rows: tuple[PositionRow, ...]) -> float:
    return math.fsum(abs(value - row.target) for value, row in zip(values, rows, strict=True)) / len(rows)


def test_generation_fits_position_rules_closer_to_the_payoffs_than_their_mean(
    declared: Declare, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    rows, rbs = strip_rows(), strip_domain(declared)
    declarer = RuleDeclarer(knowledge, rbs.context)

    result = create_value_generator().generate(rbs, rows, rows, SETTINGS, declarer)

    terms = [rule for rule in result.rules if rule.name != CONSTANT_RULE]
    assert result.context == "strip"
    assert [fit.price for fit in result.fits] == [0.1, 0.01, 0.001]
    assert result.chosen == min(result.fits, key=lambda fit: (fit.held_out_loss, fit.terms_kept))
    assert result.chosen is not None and result.chosen.terms_kept == len(terms) > 0
    assert {rule.rule for rule in terms} <= set(result.candidates)
    valued = create_rule_based_system(knowledge, rbs.context)
    values = [valued.value(row.state, row.player) for row in rows]
    mean = math.fsum(row.target for row in rows) / len(rows)
    assert mean_absolute_error(values, rows) < mean_absolute_error([mean] * len(rows), rows)
    assert any(message.startswith("Chose price ") for message in caplog.messages)
    assert any(" candidate terms after " in message for message in caplog.messages)


def test_several_targets_share_one_search_and_each_gets_its_own_fit_and_strengths(
    declared: Declare, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    rows, rbs = strip_rows(), strip_domain(declared)
    payoffs = np.array([row.target for row in rows])
    targets = {"payoff": (payoffs, payoffs), "flipped": (1.0 - payoffs, 1.0 - payoffs), "flat": (np.full(len(rows), 0.5),) * 2}

    results = create_value_generator().generate_for_targets(
        rbs, rows, rows, targets, SETTINGS, RuleDeclarer(knowledge, rbs.context)
    )

    payoff, flipped, flat = results["payoff"], results["flipped"], results["flat"]
    assert list(results) == ["payoff", "flipped", "flat"]
    assert payoff.candidates == flipped.candidates and payoff.rules and flat.rules == ()
    assert [term for term, _ in payoff.strengths] == [
        rule.rule for rule in payoff.rules if rule.name != CONSTANT_RULE
    ]
    assert all(strength != 0.0 for _, strength in payoff.strengths)
    shared = dict(payoff.strengths).keys() & dict(flipped.strengths).keys()
    assert shared and all(dict(payoff.strengths)[term] * dict(flipped.strengths)[term] < 0 for term in shared)
    assert any(message.startswith("payoff: Chose price ") for message in caplog.messages)
    assert "flat: Every training payoff is 0.5: nothing to fit" in caplog.messages


def test_a_blank_term_s_rule_adds_nothing_where_it_reads_nothing(
    declared: Declare, knowledge: KnowledgeBase
) -> None:
    rows, rbs = strip_rows(), strip_domain(declared)
    seed = Expression("(1 if wins(me) else None)", 1, 0)
    declarer = RuleDeclarer(knowledge, rbs.context)

    result = create_value_generator().generate(rbs, rows, rows, replace(SETTINGS, candidates=1), declarer, (seed,))

    # Only the seed is tried; its rule keeps the fit's weight, and a position it reads nothing in is left to the
    # constant alone.
    terms = [rule for rule in result.rules if rule.name != CONSTANT_RULE]
    ((constant,),) = ([rule for rule in result.rules if rule.name == CONSTANT_RULE],)
    assert [rule.rule for rule in terms] == [PythonRule("(1 if wins(me) else None)")]
    assert result.strengths[0][1] == pytest.approx(terms[0].weight(rbs.context))
    valued = create_rule_based_system(knowledge, rbs.context)
    added = lambda row: {rule.name: value for rule, value in valued.explain(row.state, row.player)}
    blank = next(row for row in rows if added(row)[terms[0].name] == 0.0)
    assert valued.value(blank.state, blank.player) == pytest.approx(constant.weight(rbs.context))


def test_without_held_out_rows_the_lowest_training_loss_is_chosen(
    declared: Declare, knowledge: KnowledgeBase
) -> None:
    rbs = strip_domain(declared)

    result = create_value_generator().generate(
        rbs, strip_rows(), (), SETTINGS, RuleDeclarer(knowledge, rbs.context)
    )

    assert all(fit.held_out_loss is None for fit in result.fits)
    assert result.chosen == min(result.fits, key=lambda fit: (fit.training_loss, fit.terms_kept))


def test_payoffs_that_never_vary_leave_nothing_to_fit(declared: Declare, knowledge: KnowledgeBase) -> None:
    rows, rbs = tuple(replace(row, target=0.5) for row in strip_rows()), strip_domain(declared)

    result = create_value_generator().generate(rbs, rows, (), SETTINGS, RuleDeclarer(knowledge, rbs.context))

    assert (result.rules, result.fits, result.chosen) == ((), (), None)
    assert knowledge.rules(rbs.context, ("position",)) == ()


def test_generation_needs_training_rows_and_a_price(declared: Declare, knowledge: KnowledgeBase) -> None:
    rbs = strip_domain(declared)
    declarer = RuleDeclarer(knowledge, rbs.context)

    with pytest.raises(ValueError, match="training rows"):
        create_value_generator().generate(rbs, (), (), SETTINGS, declarer)
    with pytest.raises(ValueError, match="price"):
        create_value_generator().generate(rbs, strip_rows(), (), replace(SETTINGS, prices=()), declarer)
