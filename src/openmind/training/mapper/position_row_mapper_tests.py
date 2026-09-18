import pytest

from openmind.evaluation.service.exact_search_tests import Declare, trust
from openmind.rbs.model.position_row import PositionRow
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.played_game import PlayedGame
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

#: A game of trust: A plays risky, then B punishes.
START = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A")))
AFTER_RISKY = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))
GAME = PlayedGame((), (START, AFTER_RISKY), (0.5, 0.75), (0.0, 1.0))


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
