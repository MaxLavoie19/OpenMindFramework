from collections.abc import Sequence

from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.model.played_game import PlayedGame


class PositionRowMapper:
    """Games played into the rows a heuristic is fitted on: every position a game went through, once for each player,
    valued at what that game paid that player.

    A game's result is a rough thing to value a position by — a position can be winning and still lost by the player
    who reached it — but it is the one value a game states for certain, and over many games it is what separates
    positions worth reaching from positions worth avoiding.

    A game that paid nobody — one cut short before it ended — has nothing to say and gives no rows."""

    def to_rows(self, game: RuleBasedGame, games: Sequence[PlayedGame]) -> tuple[PositionRow, ...]:
        players = game.players().names
        rows: list[PositionRow] = []
        for played in games:
            if len(played.payoffs) != len(players):
                continue
            for state in played.states:
                rows.extend(
                    PositionRow(state, player, payoff) for player, payoff in zip(players, played.payoffs, strict=True)
                )
        return tuple(rows)
