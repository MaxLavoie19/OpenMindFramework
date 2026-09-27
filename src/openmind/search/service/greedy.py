import logging

from openmind.heuristic.model.node import Node
from openmind.heuristic.service.successor_move_rater import SuccessorMoveRater
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_settings import SearchSettings
from openmind.search.model.strategy import Strategy
from openmind.world.model.action import Action

logger = logging.getLogger(__name__)


class Greedy:
    """The planner that looks one step and no further: rate this position's actions, play the best.

    **A heuristic no tree can use is a heuristic nothing can use.** Until this, the only consumer of a move
    value anywhere was the tree search's priors, and the only searchless planner read neither heuristic port —
    so everything the heuristic builder produces needed a tree before it meant anything. Some work has no tree:
    a ballistic problem is solved by calculating, not by searching, and the optimizer port says so outright.

    Two ways to answer, and the first needs no lookahead at all:

    - **Where a move value fills, the move value is already the policy.** What each action is worth to the
      player taking it, best one played. Nothing is expanded, nothing is simulated.
    - **Where only a position value fills**, each action's outcomes are valued and weighed by their
      probability, and the action whose expectation is highest is played. A finished outcome is worth what the
      game paid there and never what a model guesses, which is the one thing about a finished position that is
      not an opinion.

    Where neither fills it says None, and the agent falls back on improvising. None means it knows nothing,
    which is not the same as everything being worth zero.

    The second of those is `SuccessorMoveRater`'s whole job — what an action is worth, from what it leads to —
    so it is asked rather than walked here, through whichever valuer the guidance carried.

    **It maximises the acting player's own entry and nothing else.** No zero sum, no ownership, no notion of an
    opponent, no turns: whose entry to maximise is read from the guidance, and what the others get is their own
    business. Where more than one player acts at once, the position-value branch has nothing honest to say —
    what an action leads to depends on what the others do at the same moment, which is a question this planner
    is by definition too shallow to ask — so it declines rather than pretending the others will pass, which
    is the rater's guard and not a second copy of it here."""

    def __init__(self, successor_move_rater: SuccessorMoveRater) -> None:
        self._successors = successor_move_rater

    def plan(
        self,
        model: object,
        knowledge_base: KnowledgeBase,
        node: Node,
        guidance: Guidance,
        settings: SearchSettings,
    ) -> Strategy | None:
        """The strategy of playing the best-rated action here, or None where nothing could rate one."""
        game = node.game
        if game is None:
            return None
        players = game.players()  # type: ignore[union-attr]
        legal = game.joint_actions(node.state)  # type: ignore[union-attr]
        mine = self._mine(legal, players.names, guidance.player)
        if mine is None:
            return None
        _, actions = mine
        chosen = self._rated(node, guidance, actions) or self._followed(node, guidance, actions)
        if chosen is None:
            return None
        logger.debug("Greedy plays %s for %s", chosen.name, guidance.player)
        return Strategy.of(node.state, chosen)

    def _mine(
        self, legal: tuple[tuple[int, tuple[Action, ...]], ...], names: tuple[str, ...], player: str
    ) -> tuple[int, tuple[Action, ...]] | None:
        """Which of the actions on offer are this player's, with their place among the players."""
        at = names.index(player) if player in names else None
        if at is None:
            return None
        return next(((index, actions) for index, actions in legal if index == at and actions), None)

    def _rated(self, node: Node, guidance: Guidance, actions: tuple[Action, ...]) -> Action | None:
        """The best of those actions by what the move value says each is worth, or None where it says nothing.

        A move value is a policy already: it is asked what each action is worth to the player taking it, so
        there is nothing left to do but take the best. An action it knows nothing about is passed over rather
        than counted as worthless."""
        if guidance.move_value is None:
            return None
        model, rater = guidance.move_value
        rated = rater.rate(model, node, actions, guidance.player)
        held = [(value, action) for value, action in zip(rated, actions, strict=False) if value is not None]
        return max(held, key=lambda one: one[0])[1] if held else None

    def _followed(self, node: Node, guidance: Guidance, actions: tuple[Action, ...]) -> Action | None:
        """The action whose outcomes are worth most to this player, or None where none could be valued.

        The walk itself belongs to the successor rater, which is the same question asked as a move value:
        what is each action worth, given what the positions it leads to are worth. It is asked through the
        valuer the guidance carried rather than one this planner chose, and it is the rater that declines
        where more than one player is choosing at the same moment."""
        if guidance.position_value is None:
            return None
        model, valuer = guidance.position_value
        rated = self._successors.through(valuer, model, node, actions, guidance.player)
        held = [(value, action) for value, action in zip(rated, actions, strict=False) if value is not None]
        return max(held, key=lambda one: one[0])[1] if held else None
