from collections.abc import Sequence

from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.model.played_game import PlayedGame


class PositionRowMapper:
    """Games played into the rows a heuristic is fitted on, each position valued at what the search made of it
    where a search ran, and at what the game paid where none did.

    **The search's answer first, because that is the whole of expanding and distilling.** A search reads the
    heuristic at its leaves, looks ahead, and comes back with something better than it was given. Fitting the
    heuristic to that is teaching it to say without looking ahead what the search needed the looking ahead to
    find — and then the next search starts from the better reading. The loop only turns if the search's answer
    is kept, and it was being discarded with the tree.

    **A game's result is the fallback and is a poor target.** It is one number credited back across every
    position in the game, so a position at move twelve wears a result decided forty moves later; it is the one
    value a game states for certain, and over many games it separates positions worth reaching from positions
    worth avoiding, but it cannot tell a corner square from a count of material — which is a thing that
    happened, with the corner square carrying the heavier weight and the wrong sign.

    A game that paid nobody and searched nothing has nothing to say and gives no rows."""

    def to_rows(self, game: RuleBasedGame, games: Sequence[PlayedGame]) -> tuple[PositionRow, ...]:
        players = game.players().names
        rows: list[PositionRow] = []
        for played in games:
            paid = played.payoffs if len(played.payoffs) == len(players) else ()
            for at, state in enumerate(played.states):
                # What the search concluded here, where it acted here and valued it. The last state has no
                # decision after it, and a game played without a search has none anywhere.
                searched = played.worth[at] if at < len(played.worth) else ()
                valued = searched if len(searched) == len(players) else paid
                if not valued:
                    continue
                rows.extend(
                    PositionRow(state, player, worth) for player, worth in zip(players, valued, strict=True)
                )
        return tuple(rows)
