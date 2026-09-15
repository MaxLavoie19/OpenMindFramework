import itertools
import logging
import math
from dataclasses import replace

import numpy as np
import pytest

from openmind.inference.model.expression import Expression
from openmind.rbs.factory.rbs_factory import create_rule_valuer, create_value_generator
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.consequence_library_tests import position, strip_domain
from openmind.rule.model.python_rule import PythonRule

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


def test_generation_fits_value_rules_closer_to_the_payoffs_than_their_mean(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    rows, domain = strip_rows(), strip_domain()

    result = create_value_generator().generate(domain, rows, rows, SETTINGS)

    base, chosen = result.value_base, result.chosen
    assert (base.domain, base.low, base.high) == ("strip", 0.0, 1.0)
    assert [fit.price for fit in result.fits] == [0.1, 0.01, 0.001]
    assert chosen == min(result.fits, key=lambda fit: (fit.held_out_loss, fit.terms_kept))
    assert chosen is not None and chosen.terms_kept == len(base.rules) > 0
    assert {rule.term for rule in base.rules} <= set(result.candidates)
    valuer = create_rule_valuer(base, domain)
    values = [valuer.value(row.state)[domain.players.names.index(row.player)] for row in rows]  # type: ignore[index]
    mean = math.fsum(row.target for row in rows) / len(rows)
    assert mean_absolute_error(values, rows) < mean_absolute_error([mean] * len(rows), rows)
    assert any(message.startswith("Chose price ") for message in caplog.messages)
    assert any(" candidate terms after " in message for message in caplog.messages)


def test_several_targets_share_one_search_and_each_gets_its_own_fit_and_strengths(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    rows, domain = strip_rows(), strip_domain()
    payoffs = np.array([row.target for row in rows])
    targets = {"payoff": (payoffs, payoffs), "flipped": (1.0 - payoffs, 1.0 - payoffs), "flat": (np.full(len(rows), 0.5),) * 2}

    results = create_value_generator().generate_for_targets(domain, rows, rows, targets, SETTINGS)

    payoff, flipped, flat = results["payoff"], results["flipped"], results["flat"]
    assert list(results) == ["payoff", "flipped", "flat"]
    assert payoff.candidates == flipped.candidates and payoff.value_base.rules and flat.value_base.rules == ()
    assert [term for term, _ in payoff.strengths] == [rule.term for rule in payoff.value_base.rules]
    assert all(strength != 0.0 for _, strength in payoff.strengths)
    shared = dict(payoff.strengths).keys() & dict(flipped.strengths).keys()
    assert shared and all(dict(payoff.strengths)[term] * dict(flipped.strengths)[term] < 0 for term in shared)
    assert any(message.startswith("payoff: Chose price ") for message in caplog.messages)
    assert "flat: Every training payoff is 0.5: nothing to fit" in caplog.messages


def test_a_blank_term_s_rule_values_as_its_fit_without_moving_the_bias() -> None:
    rows, domain = strip_rows(), strip_domain()
    seed = Expression("(1 if wins(me) else None)", 1, 0)

    result = create_value_generator().generate(domain, rows, rows, replace(SETTINGS, candidates=1), (seed,))

    # Only the seed is tried: its rule's weight is the fit's weight over the scale, and a blank row adds nothing.
    base = result.value_base
    assert [rule.term for rule in base.rules] == [PythonRule("(1 if wins(me) else None)")]
    assert result.strengths[0][1] == pytest.approx(base.rules[0].weight)
    valuer = create_rule_valuer(base, domain)
    blank = next(row for row in rows if valuer.explain(row.state, row.player)[0][1] == 0.0)  # type: ignore[index]
    assert valuer.value(blank.state)[domain.players.names.index(blank.player)] == pytest.approx(  # type: ignore[index]
        1.0 / (1.0 + math.exp(-base.bias))
    )


def test_without_held_out_rows_the_lowest_training_loss_is_chosen() -> None:
    result = create_value_generator().generate(strip_domain(), strip_rows(), (), SETTINGS)

    assert all(fit.held_out_loss is None for fit in result.fits)
    assert result.chosen == min(result.fits, key=lambda fit: (fit.training_loss, fit.terms_kept))


def test_payoffs_that_never_vary_leave_nothing_to_fit() -> None:
    rows = tuple(replace(row, target=0.5) for row in strip_rows())

    result = create_value_generator().generate(strip_domain(), rows, (), SETTINGS)

    base = result.value_base
    assert (base.rules, base.low, base.high, result.fits, result.chosen) == ((), 0.5, 0.5, (), None)


def test_generation_needs_training_rows_and_a_price() -> None:
    with pytest.raises(ValueError, match="training rows"):
        create_value_generator().generate(strip_domain(), (), (), SETTINGS)
    with pytest.raises(ValueError, match="price"):
        create_value_generator().generate(strip_domain(), strip_rows(), (), replace(SETTINGS, prices=()))
