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
    #: What the search concluded each of those positions is worth, per player, where it valued one.
    #:
    #: **The search's answer is better than the heuristic that fed it, and it was being thrown away.** A
    #: position heuristic says what a board looks worth; a search reads it at the leaves, looks ahead, and
    #: comes back with something better — and until now only the move survived that. Kept, it is what the
    #: heuristic can be taught to say without the looking ahead: expand, then distil.
    #:
    #: **It is the target a payoff cannot be.** What a game paid is one number credited back across every
    #: position in it, so a position at move twelve wears the result of something decided forty moves later.
    #: This is about *this* position, there is one for every position searched, and it improves as the agent
    #: does — where an outside engine is a fixed ceiling and agreeing with it perfectly is where that stops.
    worth: tuple[tuple[State, tuple[float, ...]], ...] = ()
    #: How many times the search visited each position it explored: how much what it says there rests on.
    #:
    #: **A distribution is not evidence until you know what it is a share of.** "Six tenths of the visits went
    #: here" is a strong claim after a thousand visits and nothing after three, and the shares look identical
    #: either way. A fit taught by both at equal say learns most of its opinion from the positions the search
    #: barely looked at, since those are the many.
    visits: tuple[tuple[State, int], ...] = ()

    def visits_at(self, state: State) -> int:
        """How many times the search visited that position; nought where it never reached it."""
        for held, count in self.visits:
            if held == state:
                return count
        return 0

    def worth_at(self, state: State) -> tuple[float, ...]:
        """What the search made of that position, per player; empty where it valued none."""
        for held, values in self.worth:
            if held == state:
                return values
        return ()

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
