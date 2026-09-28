from dataclasses import dataclass

from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Studied:
    """A position of a game already played, with what the game it came from came to.

    **A position on its own has lost the two things that say what it is worth learning from.** Who played it
    and how it ended are facts about the position as much as about the game: a position from a game one
    heuristic won is evidence about that heuristic, and a position from a game that ran out of moves is a
    different thing from one that ended in a win, however alike the boards look.

    `played_with` is what each player played with, in the players' order, named by the ruleset it is — so a
    game between two heuristics says which was on which side, and a player with no heuristic is named as
    having played the game and nothing else, which is a real thing to have played with.

    `ending` is what the game said about why it stopped, and empty where it said nothing and simply left
    nobody an action. `ply` is how far into the game the position is, counting the start as nought — so the
    opening positions of a thousand games can be told from the endings of them without replaying anything."""

    state: State
    ply: int
    game: str
    players: tuple[str, ...] = ()
    played_with: tuple[str, ...] = ()
    payoffs: tuple[float, ...] = ()
    ending: str = ""

    @property
    def decisive(self) -> bool:
        """Whether anyone came out ahead of the others in the game this came from."""
        return bool(self.payoffs) and len(set(self.payoffs)) > 1

    def won_by(self) -> str:
        """What the winner played with, or empty where nobody won or nobody said what they played with."""
        if not self.decisive or len(self.played_with) != len(self.payoffs):
            return ""
        best = max(range(len(self.payoffs)), key=lambda at: self.payoffs[at])
        return self.played_with[best]
