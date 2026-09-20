from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State

#: What to play in one state: each action with the probability of playing it, summing to 1.
type MoveDistribution = tuple[tuple[Action, float], ...]


@dataclass(frozen=True, slots=True)
class Strategy:
    """A mixed strategy: a move distribution per state. In this state, play one of these actions at these
    probabilities; in that one, play one of those instead.

    Planning gives a strategy rather than a plan. A plan is a series of actions and states, and it solves nothing a
    strategy doesn't solve better: a strategy covers the responses too, and the paths that invalidate it. The plan can
    still be read off it — the line where everyone plays as expected — and a state the strategy says nothing about is
    where the agent strategizes again."""

    moves: tuple[tuple[State, MoveDistribution], ...]

    def at(self, state: State) -> MoveDistribution:
        """What to play in that state; empty where the strategy says nothing about it, which is where to strategize
        again."""
        for held, distribution in self.moves:
            if held == state:
                return distribution
        return ()

    def chosen(self, state: State) -> Action | None:
        """The likeliest action in that state, ties going to the first; None where the strategy says nothing."""
        distribution = self.at(state)
        return max(distribution, key=lambda pair: pair[1])[0] if distribution else None

    def states(self) -> tuple[State, ...]:
        return tuple(state for state, _ in self.moves)

    @staticmethod
    def of(state: State, action: Action) -> "Strategy":
        """The strategy playing that one action in that state: what a search finding one move gives."""
        return Strategy(((state, ((action, 1.0),)),))
