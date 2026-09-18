import math
from collections.abc import Sequence

from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.model.position_row import PositionRow
from openmind.training.constant.training_constant import OUTCOME_TARGET, SEARCH_TARGET, VALUE_TARGETS
from openmind.training.model.played_game import PlayedGame
from openmind.world.service.state_reader import StateReader


class PositionRowMapper:
    """Maps self-play games to the position rows value rules are fitted on. With the outcome target, every position of a
    game gives a row for every player, valued at that player's final payoff; with the search target, a row for the
    player to act, valued at the search's mean payoff there, and none where nothing was searched, such as a random move
    played out of time."""

    def __init__(self, state_reader: StateReader) -> None:
        self._state_reader = state_reader

    def to_rows(self, rbs: RuleBasedSystem, games: Sequence[PlayedGame], target: str) -> tuple[PositionRow, ...]:
        if target not in VALUE_TARGETS:
            raise ValueError(f"Unknown value target {target!r}: expected one of {', '.join(VALUE_TARGETS)}")
        names = rbs.players().names
        rows: list[PositionRow] = []
        for game in games:
            if target == OUTCOME_TARGET:
                for state in game.states:
                    rows.extend(PositionRow(state, name, payoff) for name, payoff in zip(names, game.payoffs, strict=True))
            elif target == SEARCH_TARGET:
                for state, value in zip(game.states, game.search_values, strict=True):
                    if math.isnan(value):
                        continue
                    rows.append(PositionRow(state, names[self._state_reader.player_to_act(state, rbs.players())], value))
        return tuple(rows)
