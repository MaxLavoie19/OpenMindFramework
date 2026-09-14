import pytest

from openmind.evaluation.service.exact_search_tests import trust
from openmind.rbs.model.position_row import PositionRow
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.played_game import PlayedGame
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

#: A game of trust: A plays risky, then B punishes.
AFTER_RISKY = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))
GAME = PlayedGame((), (trust().initial_state, AFTER_RISKY), (0.5, 0.75), (0.0, 1.0))


def test_the_outcome_target_values_every_position_for_every_player_at_their_final_payoff() -> None:
    domain = trust()

    rows = PositionRowMapper(StateReader()).to_rows(domain, (GAME,), "outcome")

    assert rows == (
        PositionRow(domain.initial_state, "A", 0.0),
        PositionRow(domain.initial_state, "B", 1.0),
        PositionRow(AFTER_RISKY, "A", 0.0),
        PositionRow(AFTER_RISKY, "B", 1.0),
    )


def test_the_search_target_values_every_position_for_the_player_to_act_at_the_search_value() -> None:
    domain = trust()

    rows = PositionRowMapper(StateReader()).to_rows(domain, (GAME,), "search")

    assert rows == (PositionRow(domain.initial_state, "A", 0.5), PositionRow(AFTER_RISKY, "B", 0.75))


def test_an_unknown_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="outcome, search"):
        PositionRowMapper(StateReader()).to_rows(trust(), (), "guess")
