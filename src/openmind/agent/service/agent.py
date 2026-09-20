import logging
from collections.abc import Sequence

from openmind.agent.service.actor import Actor
from openmind.budget.model.allocation import Allocation
from openmind.budget.model.budget import Budget
from openmind.budget.model.time_manager import TimeManager
from openmind.knowledge.constant.task_constant import MOVE_VALUE, PLANNING, POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.search.model.planner import Planner
from openmind.search.model.strategy import Strategy
from openmind.world.service.world import World

logger = logging.getLogger(__name__)

#: The tasks a move needs: what to plan with, and the heuristics the planner reads.
MOVE_TASKS = (PLANNING, POSITION_VALUE, MOVE_VALUE)


class Agent:
    """The loop: perceive, process, plan, communicate, act, under one budget.

    The planner strategizes while the actor acts on what it has so far. The state is the world's, so the actor answers
    what is actually happening and the planner works from where the game really is; where the strategy has nothing for
    the current state, the actor waits and the planner strategizes from there.

    What each step runs with is the time management policy's: which model fills each task, and how far the planner may
    explore."""

    def __init__(self, time_manager: TimeManager[object], planner: Planner[object], actor: Actor | None = None) -> None:
        self._time_manager = time_manager
        self._planner = planner
        self._actor = actor

    def allocate(
        self, knowledge_base: KnowledgeBase, game: RuleBasedGame, world: World, budget: Budget, tasks: Sequence[str] = MOVE_TASKS
    ) -> Allocation:
        """What this step runs with, as the time management policy says."""
        return self._time_manager.manage(None, knowledge_base, game.node(world.current()), budget, tasks)

    def play(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        world: World,
        guidance: Guidance,
        budget: Budget,
    ) -> Strategy | None:
        """Strategizes from where the world stands, acting on the strategy as it is worked out. Gives the strategy, or
        None where nothing could be worked out."""
        allocation = self.allocate(knowledge_base, game, world, budget)
        node = game.node(world.current())
        strategy = self._planner.plan(None, knowledge_base, node, guidance, allocation.settings)
        if self._actor is not None:
            self._actor.follow(strategy)
            self._actor.act(world)
        if strategy is None:
            logger.info("Nothing to play for %s here", guidance.player)
        return strategy
