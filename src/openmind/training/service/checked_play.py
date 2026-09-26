import logging
from collections.abc import Sequence
from dataclasses import dataclass

from typing import Protocol

from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Deciding(Protocol):
    """Whatever decides whether a move was really allowed, and what it leads to.

    **Two questions and not a whole game, because the thing that decides need not be one.** A learner and the
    game it learned from may not even say a position the same way: chess's own engine holds a board, the
    declared game holds two grids of pieces and colours, and the learning side holds one grid of records. A
    decider asked for a `RuleBasedGame` can only be a game built OMF's way, which rules out the one decider
    that is never wrong — the thing being learned from.

    So what is asked for is the judgement, and whoever can give it may. An integrator wraps its own engine in
    fifteen lines; a declared game already answers both; another learner answers as well as it has learned."""

    def allows(self, state: State, action: Action) -> bool: ...

    def leads_to(self, state: State, action: Action) -> State | None: ...


class DecidingGame:
    """A game OMF holds, answering as a decider: what its rules allow, and the likeliest position a move leads to."""

    def __init__(self, game: RuleBasedGame) -> None:
        self._game = game

    def allows(self, state: State, action: Action) -> bool:
        return self._game.allows(state, action, self._game.acting_player(state))

    def leads_to(self, state: State, action: Action) -> State | None:
        outcomes = self._game.outcomes(state, action).outcomes
        return max(outcomes, key=lambda one: one[1])[0] if outcomes else None


@dataclass(frozen=True, slots=True)
class Checked:
    """How a game of believed rules fared against a game that decides: the positions it reached, the moves it
    played, and the first move it proposed that was not allowed.

    `proposed` is what it offered and `at` the position it offered it in, both None where it never proposed
    anything impossible. `plies` is how far it got before that happened, which is the number worth watching:
    a believed game that lasts forty moves believes something close to the real thing, and one that falls over
    on its first move does not."""

    states: tuple[State, ...]
    played: tuple[Action, ...]
    proposed: Action | None = None
    at: State | None = None
    refused: tuple[Action, ...] = ()

    @property
    def plies(self) -> int:
        return len(self.played)

    @property
    def held_up(self) -> bool:
        """Whether every move it proposed was one the deciding game allowed."""
        return self.proposed is None


class CheckedPlay:
    """Plays a game whose rules are believed, against a game that decides whether each move was really allowed.

    **A search over rules that let too much through does not go wrong gracefully — it goes looking.** The moves
    a believed game offers that the real one would refuse are the cheapest moves in its tree, because nothing
    answers them; a planner rewarded for finding what nobody can punish will find exactly those and prefer
    them. Playing such a game against nothing produces a heuristic fitted on lines that cannot happen.

    So every move is put to the deciding game before it is played. What it allows is played; what it refuses
    ends the game there and is kept, because a move proposed and refused is the one thing a learner most wants
    and the one thing walking at random almost never produces — a candidate the believed rules got wrong,
    found by believing them.

    Nothing here is one game or two: a believed game and a deciding game are both games, and which is which is
    the caller's. The same shape measures a learner against the thing it learned from, one agent's model of
    another against that other, and a relaxation against the game it was relaxed from.

    It keeps nothing: built once, it is given both games on every call."""

    def play(self, believed: RuleBasedGame, deciding: Deciding, chosen, steps: int | None = None) -> Checked:
        """One game of the believed rules, checked move by move, stopping where the deciding game refuses.

        `chosen` is asked for the move to play in a position, given the believed game and that position, and
        gives None where it has nothing to play. It is handed in rather than held so that this measures a
        believed game and not a way of choosing."""
        state = believed.start()
        states: list[State] = [state]
        played: list[Action] = []
        while steps is None or len(played) < steps:
            # A position the believed rules leave nobody a move in is a game over as far as they know, which
            # is how a game with no ending rule ends. Asking who acts there raises, and rightly.
            if not believed.joint_actions(state):
                break
            action = chosen(believed, state)
            if action is None:
                break
            if not deciding.allows(state, action):
                logger.info(
                    "The believed rules proposed %s after %d moves, which the game does not allow",
                    action.parameters,
                    len(played),
                )
                return Checked(tuple(states), tuple(played), action, state, (action,))
            reached = deciding.leads_to(state, action)
            if reached is None:
                break
            state = reached
            states.append(state)
            played.append(action)
        logger.info("The believed rules played %d moves, every one of them allowed", len(played))
        return Checked(tuple(states), tuple(played))

    def furthest(self, checked: Sequence[Checked]) -> int:
        """The most moves any of those games lasted before proposing something impossible."""
        return max((one.plies for one in checked), default=0)
