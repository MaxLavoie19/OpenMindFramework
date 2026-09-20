import logging
import threading
from collections.abc import Callable
from dataclasses import replace

from openmind.abstraction.model.abstractor import Abstractor
from openmind.agent.model.delegation import Delegation
from openmind.agent.model.report import Report
from openmind.agent.service.agent import Agent
from openmind.budget.model.budget import Budget
from openmind.heuristic.model.node import Node
from openmind.knowledge.constant.knowledge_constant import DONE, RUNNING
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.task import Task
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.wall_time_source import WallTimeSource
from openmind.utility.service.utility import Utility
from openmind.world.model.state import State
from openmind.world.service.world import World

logger = logging.getLogger(__name__)

#: How long a level waits before looking at its world again, where nothing changed while it played, in seconds.
LOOK_AGAIN_SECONDS = 0.01

#: The variable a belief holds what pursuing a goal in a level was worth under.
WORTH = "what {goal} in {context} was worth"

#: The variable a belief holds what pursuing a goal in a level takes, in seconds.
SECONDS = "seconds {goal} in {context} takes"


class Level:
    """One level of the hierarchy: a context, the game it plays there, the world it sees, the models it runs with, and
    the agent loop running in it.

    Its world is an abstraction of the world everything shares, made by a model of the abstraction task: a level holds
    the detail it works on and none of the rest, and fetches what it doesn't hold by acting. Where it has no
    abstraction model, it sees the state as it is.

    A parent runs it by delegating a goal and a budget. It pursues that goal in its own context, with its own models —
    a level is solved by what suits it, minimax for tic-tac-toe, a Monte-Carlo tree search for chess — and reports
    back where it left its level, what that was worth and what it spent. What it did is kept as a task in its own
    context, so what delegating to it brings can be looked back on."""

    def __init__(
        self,
        context: str,
        game: RuleBasedGame,
        agent: Agent,
        guidance: Guidance,
        world: World,
        abstractor: tuple[object, Abstractor[object]] | None = None,
        utility: Utility | None = None,
        time_source: TimeSource | None = None,
        wait_seconds: float = LOOK_AGAIN_SECONDS,
    ) -> None:
        self._context = context
        self._game = game
        self._agent = agent
        self._guidance = guidance
        self._world = world
        self._abstractor = abstractor
        self._utility = utility
        self._time_source = WallTimeSource() if time_source is None else time_source
        self._wait = wait_seconds
        self._idle = threading.Event()

    @property
    def context(self) -> str:
        """The context id this level is a level of."""
        return self._context

    @property
    def world(self) -> World:
        """The world as this level sees it."""
        return self._world

    def perceived(self, node: Node) -> State:
        """Takes what the shared world now says, as this level sees it: its abstraction model's answer, or the state
        as it is where it has none or where the model says nothing about this state."""
        if self._abstractor is None:
            return self._world.perceived(node.state)
        model, abstractor = self._abstractor
        seen = abstractor.abstract(model, node, self._context)
        return self._world.perceived(node.state if seen is None else seen)

    def run(
        self,
        knowledge_base: KnowledgeBase,
        delegation: Delegation,
        stopping: threading.Event | None = None,
        informing: Callable[[Node], State] | None = None,
    ) -> Report:
        """Pursues the delegated goal until the level ends, the budget runs out or it is stopped; what came of it.

        It plays its own agent loop: strategizing and acting as one step, over and over, on the world as it stands.
        Where a step changed nothing — its level waits on what performs its actions — it waits before looking again,
        rather than spinning.

        `informing` is told where the level stands after every step: the parent is kept informed as the work goes,
        rather than only at the end, and what it makes of what it is told is its own abstraction's. A coach whose
        child plays the moves hears of each one as it is played, in time to comment on it."""
        started = self._time_source.now()
        guidance = self._planning_for(delegation)
        task = self._started(knowledge_base, delegation)
        reached = False
        while True:
            if self._over(self._world.current()):
                reached = True
                break
            left = delegation.budget.seconds - (self._time_source.now() - started)
            if left <= 0 or (stopping is not None and stopping.is_set()):
                break
            changed = self._world.changes()
            self._agent.play(
                knowledge_base, self._game, self._world, guidance, Budget(left, delegation.budget.of_clock)
            )
            self._informs(informing, self._world.current())
            if self._world.changes() == changed:
                self._waited(stopping)
        state = self._world.current()
        report = Report(
            delegation,
            state,
            self._worth(knowledge_base, state, guidance.player),
            self._time_source.now() - started,
            reached,
            task.id,
        )
        self._done(knowledge_base, task, report)
        logger.info(
            "%s %s after %.3g seconds, worth %s",
            self._readable(knowledge_base, delegation),
            "reached" if reached else "ran out",
            report.seconds,
            "nothing known" if report.value is None else f"{report.value:.4g}",
        )
        return report

    def _informs(self, informing: Callable[[Node], State] | None, state: State) -> None:
        """Tells whoever is being kept informed where the level stands, as a node of this level's game: what they make
        of it is theirs. A parent that isn't there to be told is no reason for the child to stop."""
        if informing is not None:
            informing(self._game.node(state))

    def _over(self, state: State) -> bool:
        """Whether the level is done with: its game says it ended, or no player has a legal action left there. A game
        that doesn't say why it ended is over when there is nothing left to play, which is what a planner reads too."""
        return self._game.ended(state) is not None or not self._game.joint_actions(state)

    def _planning_for(self, delegation: Delegation) -> Guidance:
        """The models this level runs with, for the player the parent delegated as."""
        if not delegation.player or delegation.player == self._guidance.player:
            return self._guidance
        return replace(self._guidance, player=delegation.player)

    def _waited(self, stopping: threading.Event | None) -> None:
        """Waits before looking again, and no longer than that: a stop asked for meanwhile ends the wait at once."""
        (self._idle if stopping is None else stopping).wait(self._wait)

    def _worth(self, knowledge_base: KnowledgeBase, state: State, player: str) -> float | None:
        """What the level it left is worth to the player, by this level's own utility; None where nothing values it."""
        if self._utility is None:
            return None
        try:
            payoff = self._game.players().payoff
        except ValueError:
            return None
        return self._utility.value(knowledge_base, self._game.node(state), state, player, payoff)

    def _started(self, knowledge_base: KnowledgeBase, delegation: Delegation) -> Task:
        """The task the run is kept under, as it starts: what it is expected to take is what it was given."""
        return knowledge_base.task(
            Task(
                self._readable(knowledge_base, delegation),
                delegation.context,
                (),
                Belief(self._variable(SECONDS, knowledge_base, delegation), delegation.context, delegation.budget.seconds),
                status=RUNNING,
                tags=(("goal", delegation.goal),),
            )
        )

    def _done(self, knowledge_base: KnowledgeBase, task: Task, report: Report) -> Task:
        """The task as the run left it: what it was actually worth, where anything could value it. Drifting that value
        over the runs is the next-best-task loop's."""
        delegation = report.delegation
        worth = (
            ()
            if report.value is None
            else (Belief(self._variable(WORTH, knowledge_base, delegation), delegation.context, report.value),)
        )
        return knowledge_base.task(replace(task, status=DONE, value=worth))

    def _variable(self, pattern: str, knowledge_base: KnowledgeBase, delegation: Delegation) -> str:
        return pattern.format(
            goal=self._goal(knowledge_base, delegation), context=knowledge_base.readable_context(delegation.context)
        )

    def _readable(self, knowledge_base: KnowledgeBase, delegation: Delegation) -> str:
        return f"{self._goal(knowledge_base, delegation)} in {knowledge_base.readable_context(delegation.context)}"

    def _goal(self, knowledge_base: KnowledgeBase, delegation: Delegation) -> str:
        goal = knowledge_base.goal_by_id(delegation.goal)
        return delegation.goal if goal is None else goal.name
