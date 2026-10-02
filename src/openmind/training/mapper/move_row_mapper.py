from collections.abc import Sequence

from openmind.rbs.model.move_row import MoveRow
from openmind.training.model.played_game import PlayedGame


class MoveRowMapper:
    """Games played into the rows a move heuristic is fitted on: each action a search settled on, with the
    chance it settled on it.

    **This is the distillation that makes the next search cheaper rather than only better.** Every other thing
    learned here improves what the agent thinks of a position. A move heuristic changes what a search *costs*:
    ordering the moves in a position takes one reading where valuing the position each move leads to takes one
    per move — about thirty-five in chess. The budget saved goes into depth, a deeper search is a better thing
    to distil, and the next move heuristic is better still. That is the loop turning rather than improving.

    **Only where a search ran.** A game somebody else played says which move was made and nothing about what
    the others were worth, and the move played is one bit where a distribution is a shape. Nothing is invented
    for those: a mapper that spread the played move's certainty over the rest would be teaching a guess.

    A position gives as many rows as the search had moves to weigh in it."""

    def to_rows(self, games: Sequence[PlayedGame]) -> tuple[MoveRow, ...]:
        """Every action a search weighed, as a row worth what the search gave it.

        The player is whoever acted there, read off the game's own action, so a row is about the side that had
        the choice rather than about whoever the game was recorded for."""
        rows: list[MoveRow] = []
        for played in games:
            for at, settled in enumerate(played.chosen):
                if not settled or at >= len(played.states) or at >= len(played.actions):
                    continue
                acting = played.actions[at].actions
                if not acting:
                    continue
                player = acting[0][0]
                for action, chance in settled:
                    rows.append(MoveRow(played.states[at], action, player, float(chance)))  # type: ignore[arg-type]
        return tuple(rows)
