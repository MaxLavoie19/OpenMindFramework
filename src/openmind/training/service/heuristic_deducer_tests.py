import logging
from dataclasses import replace

import numpy as np
import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.mechanics_tests import new_mechanics
from openmind.inference.service.position_view_tests import capture_domain, capture_position
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.training.model.signal_settings import SignalSettings
from openmind.training.service.heuristic_deducer import HeuristicDeducer
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players

pytestmark = pytest.mark.log_level("INFO")


def new_deducer() -> HeuristicDeducer:
    return HeuristicDeducer(ExpressionGenerator(VariableNameMapper()), new_mechanics())


def test_a_base_naming_players_gives_every_principle_with_its_premises(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")

    signals = new_deducer().deduce(capture_domain())

    assert [(signal.name, signal.premises) for signal in signals] == [
        ("options", ()),
        ("my options", ("options",)),
        ("their options", ("options",)),
        ("goal distance", ()),
        ("owned cell", ("options",)),
        ("taking cell", ("owned cell",)),
        ("losing cell", ("owned cell",)),
        ("taking moves cell", ("taking cell", "their options")),
        ("losing moves cell", ("losing cell", "my options")),
        ("fork cell", ("taking cell",)),
        ("material cell", ("owned cell", "taking moves cell")),
        ("hanging cell", ("losing cell",)),
    ]
    assert (
        "Derived options from the rules: ((mine - theirs) / (mine + theirs) if (mine := here.mobility(me)) + "
        "(theirs := here.mobility(other)) else 0.0)"
    ) in caplog.messages
    assert any(message.startswith("Derived fork cell from taking cell: ") for message in caplog.messages)
    # Alone, a piece has one move on either end of the row and two on the 5 cells between: 12 moves over 7 cells.
    assert "Worth of a thing of cell by its kind (), its average legal moves alone on every cell: {(): 1.714}" in caplog.messages


def test_the_signals_read_what_they_say_and_are_blank_when_it_isn_t_there() -> None:
    domain = capture_domain()
    signals = new_deducer().deduce(domain)
    # A at 2 and 6, B at 4 and 7, A to act: A has 4 moves and B 3. A's 6 takes 7, leaving B 2 moves; B's 7 takes 6,
    # leaving A 2 moves of 4, and nothing of A's could take on 6: it hangs. Only moving 2 to 3 leaves two takings, 3
    # taking 4 and 6 taking 7. At the start nothing can be taken, now or after one move twice.
    rows = (PositionRow(capture_position({2: "A", 6: "A", 4: "B", 7: "B"}, "A"), "A", 0.0), PositionRow(domain.initial_state, "A", 0.0))

    columns = new_evaluator().columns(domain, rows, [signal.source for signal in signals])  # type: ignore[misc]

    readings = {signal.name: column.tolist() for signal, column in zip(signals, columns, strict=True)}  # type: ignore[union-attr]
    # Normalized: 4 and 3 moves read (4 - 3) / 7; one of B's 2 pieces taken reads 1/2; B losing 1 of its 3 moves 1/3; A
    # losing 2 of its 4 moves -1/2; one forking move of A's 4 reads 1/4; one of A's 2 pieces hanging -1/2.
    assert {name: values[0] for name, values in readings.items() if name != "goal distance"} == pytest.approx(
        {
            "options": 1 / 7,
            "my options": 4 / 7,
            "their options": -3 / 7,
            "owned cell": 0.0,
            "taking cell": 0.5,
            "losing cell": -0.5,
            "taking moves cell": 1 / 3,
            "losing moves cell": -0.5,
            "fork cell": 0.25,
            "material cell": 0.0,
            "hanging cell": -0.5,
        }
    )
    assert [readings[name][1] for name in ("options", "my options", "their options", "owned cell", "material cell")] == [0.0, 0.5, -0.5, 0.0, 0.0]
    blank = ("taking cell", "losing cell", "taking moves cell", "losing moves cell", "fork cell", "hanging cell")
    assert all(np.isnan(readings[name][1]) for name in blank)
    # Neither player can take the other's last piece within 2 moves here, nor at the start.
    assert np.isnan(readings["goal distance"][0]) and np.isnan(readings["goal distance"][1])


def test_goal_distance_reads_who_is_closer_to_winning_within_the_limit() -> None:
    domain = capture_domain()
    # A at 2 and 3, B at 4: A takes B's last piece in 1 move; B needs 2, taking 3 then 2.
    close = PositionRow(capture_position({2: "A", 3: "A", 4: "B"}, "A"), "A", 0.0)
    other = PositionRow(close.state, "B", 0.0)
    (near,) = [signal for signal in new_deducer().deduce(domain) if signal.name == "goal distance"]
    (far,) = [signal for signal in new_deducer().deduce(domain, goal_limit=1) if signal.name == "goal distance"]

    (column,) = new_evaluator().columns(domain, (close, other), [near.source])  # type: ignore[list-item]
    (limited,) = new_evaluator().columns(domain, (close, other), [far.source])  # type: ignore[list-item]

    assert list(column) == pytest.approx([1 / 3, -1 / 3])  # type: ignore[arg-type]
    # Within 1 move, B can't win: its distance is the limit plus one, 2.
    assert list(limited) == pytest.approx([1 / 3, -1 / 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="goal limit of 1 or more"):
        SignalSettings(goal_limit=0)


def test_a_domain_whose_initial_position_names_no_player_gives_only_the_options_and_goal_distance() -> None:
    signals = new_deducer().deduce(create_tictactoe_domain())

    assert [signal.name for signal in signals] == ["options", "my options", "their options", "goal distance"]


def test_a_domain_without_two_players_raises() -> None:
    alone = replace(capture_domain(), players=Players(("A",), "turn", ("payoff(A)",)))

    with pytest.raises(ValueError, match="between two players, not 1"):
        new_deducer().deduce(alone)
