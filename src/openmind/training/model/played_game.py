from dataclasses import dataclass

from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class PlayedGame:
    """A game played out: every position it went through, in order, what the players did in each, what it paid them at
    the end, and why it ended.

    `payoffs` are in the order of the players' names, and empty where the game ended without paying anyone — a game
    cut short by the steps it was given. `ending` is what the game says about why it ended, None where it says
    nothing and the game simply left nobody an action.

    Its two seeds are kept apart: `agent_seed` is what the players' mixed strategies were drawn from and
    `outcome_seed` what the game's own chances were. Apart, a game can be played again from its actions alone — the
    same chances come up in the same places — which is how it is shown again without keeping every position."""

    states: tuple[State, ...]
    actions: tuple[JointAction, ...]
    payoffs: tuple[float, ...] = ()
    ending: str | None = None
    agent_seed: int | None = None
    outcome_seed: int | None = None
    #: What the search concluded each position it acted in was worth, per player, in step with `actions`.
    #:
    #: **Expand, then distil.** A search reads the heuristic at its leaves, looks ahead, and comes back with a
    #: better answer than the heuristic gave it. That answer is what the heuristic can be taught to say
    #: without looking ahead — and it was being discarded with the tree the moment a move was picked.
    #:
    #: **And it is the target a payoff cannot be.** What a game paid is one number credited back across every
    #: position in it, so a position at move twelve wears a result decided forty moves later; fitted to that,
    #: a corner square and a count of material are nearly indistinguishable, which is a thing that happened.
    #: This is about *this* position, there is one wherever a search ran, and it rises as the agent does.
    #:
    #: Empty where nothing searched — a game played by something that does not look ahead says nothing here.
    worth: tuple[tuple[float, ...], ...] = ()

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def decisive(self) -> bool:
        """Whether anyone came out ahead: a game everyone was paid the same is not one anything was learned from
        about how to play better."""
        return bool(self.payoffs) and len(set(self.payoffs)) > 1
