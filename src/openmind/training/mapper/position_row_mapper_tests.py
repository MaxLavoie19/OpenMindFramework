import pytest

from openmind.rbs.model.position_row import PositionRow
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.consequence_library_tests import Declare
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.structure.model.map import Map
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.played_game import PlayedGame
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

#: A game of trust: A plays risky, then B punishes.
START = State.of(payoff=Map.of({"A": None, "B": None}), turn="A")
AFTER_RISKY = State.of(payoff=Map.of({"A": None, "B": None}), turn="B")
GAME = PlayedGame((), (START, AFTER_RISKY), (0.5, 0.75), (0.0, 1.0))


def trust(declared: Declare) -> RuleBasedSystem:
    """A plays safe (0.5 each) or risky; after risky, B plays punish (A 0, B 1) or reward (A 1, B 0)."""
    unset = PythonRule("payoff['A'] is None")

    def payoffs(a: float, b: float) -> PythonRule:
        return PythonRule(f"payoff = payoff.with_item('A', {a!r}).with_item('B', {b!r})")

    a_moves, b_moves = (unset, PythonRule("turn == 'A'")), (unset, PythonRule("turn == 'B'"))
    return declared(
        START,
        legal={"safe": a_moves, "risky": a_moves, "punish": b_moves, "reward": b_moves},
        outcomes={
            "safe": ((1.0, payoffs(0.5, 0.5)),),
            "risky": ((1.0, PythonRule("turn = 'B'")),),
            "punish": ((1.0, payoffs(0.0, 1.0)),),
            "reward": ((1.0, payoffs(1.0, 0.0)),),
        },
        players=Players(("A", "B"), "turn", "payoff"),
        context="trust",
    )


def test_the_outcome_target_values_every_position_for_every_player_at_their_final_payoff(declared: Declare) -> None:
    rbs = trust(declared)

    rows = PositionRowMapper(StateReader()).to_rows(rbs, (GAME,), "outcome")

    assert rows == (
        PositionRow(START, "A", 0.0),
        PositionRow(START, "B", 1.0),
        PositionRow(AFTER_RISKY, "A", 0.0),
        PositionRow(AFTER_RISKY, "B", 1.0),
    )


def test_the_search_target_values_every_position_for_the_player_to_act_at_the_search_value(declared: Declare) -> None:
    rbs = trust(declared)

    rows = PositionRowMapper(StateReader()).to_rows(rbs, (GAME,), "search")

    assert rows == (PositionRow(START, "A", 0.5), PositionRow(AFTER_RISKY, "B", 0.75))


def test_an_unknown_target_is_rejected(declared: Declare) -> None:
    with pytest.raises(ValueError, match="outcome, search"):
        PositionRowMapper(StateReader()).to_rows(trust(declared), (), "guess")
