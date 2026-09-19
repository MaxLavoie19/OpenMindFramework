import json
import logging
from collections.abc import Sequence

from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.knowledge.constant.rule_kind_constant import PICTURE, RECORD
from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.rbs.constant.game_record_constant import ACTIONS, PAYOFFS
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.heuristic.model.node import Node
from openmind.heuristic.service.rule_heuristic import RuleHeuristic
from openmind.rbs.service.simulation import Simulation
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.players import Players
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class RuleBasedGame:
    """A temporary facade over a game's rule-based systems, for the packages not yet reworked into stateless services:
    the simulation ruleset's RBS, run by the simulation service, and the heuristics' RBSs, one per ruleset, each rule
    weighed by its own ruleset's weight. Each package drops it at its own step (see `doc/interfaces/game.md`).

    All players play at the same time, all the time: the players acting in a state are those with a legal action
    there, and a game played in turns leaves a player no action outside their turn."""

    def __init__(
        self,
        context: str,
        simulation_rbs: RuleBasedSystem | None,
        heuristics: tuple[RuleBasedSystem, ...],
        simulation: Simulation,
        heuristic: RuleHeuristic,
        context_id: str | None = None,
    ) -> None:
        self._context = context
        self._context_id = context if context_id is None else context_id
        self._game = simulation_rbs
        self._heuristics = heuristics
        self._simulation = simulation
        self._heuristic = heuristic
        systems = (() if simulation_rbs is None else (simulation_rbs,)) + heuristics
        self._rules = tuple(rule for system in systems for rule, _ in system.rules)
        self._position = next((system for system in heuristics if system.ruleset.task == POSITION_VALUE), None)
        self._move = next((system for system in heuristics if system.ruleset.task == MOVE_VALUE), None)
        self._start: State | None = None
        self._players: Players | None = None

    @property
    def context_id(self) -> str:
        """The id the knowledge base links the context by."""
        return self._context_id

    @property
    def context(self) -> str:
        """The game the rules were retrieved for; a relaxation's context is its own."""
        return self._context

    @property
    def simulation_rbs(self) -> RuleBasedSystem:
        """The simulation ruleset's RBS; a context without one raises ValueError."""
        if self._game is None:
            raise ValueError(f"{self._context} has no rule saying where the game starts")
        return self._game

    @property
    def rules(self) -> tuple[RuleRecord, ...]:
        """Every rule its RBSs run, the simulation's first."""
        return self._rules

    def weight(self, rule: RuleRecord) -> float:
        """What a heuristic rule weighs in its ruleset; 0 for a rule none of its heuristics lists."""
        for system in (self._position, self._move):
            if system is not None and system.weight(rule):
                return system.weight(rule)
        return 0.0

    def start(self) -> State:
        """Where the game starts; read once."""
        if self._start is None:
            self._start = self._simulation.start(self.simulation_rbs)
        return self._start

    def players(self) -> Players:
        """Who plays, and the Map holding each player's payoff; read once."""
        if self._players is None:
            self._players = self._simulation.players(self.simulation_rbs)
        return self._players

    def actions(self, state: State, limit: int | None = None, player: str | None = None) -> tuple[Action, ...]:
        return self._simulation.actions(self.simulation_rbs, state, limit, player)

    def actions_with_statistics(
        self, state: State, limit: int | None = None, player: str | None = None
    ) -> tuple[tuple[Action, ...], SolveStatistics]:
        return self._simulation.actions_with_statistics(self.simulation_rbs, state, limit, player)

    def acting(self, state: State) -> tuple[int, ...]:
        return self._simulation.acting(self.simulation_rbs, state)

    def acting_player(self, state: State) -> str:
        return self._simulation.acting_player(self.simulation_rbs, state)

    def joint_actions(self, state: State) -> tuple[tuple[int, tuple[Action, ...]], ...]:
        return self._simulation.joint_actions(self.simulation_rbs, state)

    def outcomes(self, state: State, action: Action) -> OutcomeDistribution:
        return self._simulation.outcomes(self.simulation_rbs, state, action)

    def joint_outcomes(self, state: State, joint: JointAction) -> OutcomeDistribution:
        return self._simulation.joint_outcomes(self.simulation_rbs, state, joint)

    def ended(self, state: State) -> str | None:
        return None if self._game is None else self._simulation.ended(self._game, state)

    def record(self, actions: Sequence[Action], payoffs: Sequence[float] | None = None) -> str | None:
        """The game's record, read from where the game starts and the actions played. The rule is also given the final
        `payoffs` (None when unknown). None where the game doesn't record itself."""
        if self._game is None or not self._game.of(RECORD):
            return None
        parameters = {ACTIONS: tuple(actions), PAYOFFS: None if payoffs is None else tuple(payoffs)}
        value = self._simulation.call(self._game, RECORD, self.start(), parameters)
        return None if value is None else str(value)

    def picture(self, state: State, **parameters: object) -> str | None:
        """The position as an SVG image, or None where the game doesn't draw itself."""
        if self._game is None:
            return None
        return self._simulation.call(self._game, PICTURE, state, parameters)  # type: ignore[return-value]

    def node(self, state: State) -> Node:
        """A node for that state of this game: what a heuristic is given, its features extracted through the game."""
        return Node(state, self)

    def value(self, state: State, player: str) -> float | None:
        """What the position is worth to the player, by its position value RBS; None where it has none, or where no
        rule could be read."""
        return None if self._position is None else self._heuristic.value(self._position, self.node(state), player)

    def values(self, state: State) -> tuple[float, ...] | None:
        """Each player's value, in the order of the players' names; None where any of them can't be valued."""
        if self._position is None:
            return None
        return self._heuristic.values(self._position, self.node(state), self.players().names)

    def rate(self, state: State, actions: tuple[Action, ...], player: str | None = None) -> tuple[float | None, ...]:
        """What each move is worth to the player taking it, by its move value RBS. Without a player, the one player
        acting in the state."""
        if self._move is None:
            return (None,) * len(actions)
        acting = self.acting_player(state) if player is None else player
        return self._heuristic.rate(self._move, self.node(state), actions, acting)

    def explain(self, state: State, player: str) -> tuple[tuple[RuleRecord, float], ...]:
        """Each position rule with what it adds to the player's value: its weight times its reading."""
        return () if self._position is None else self._heuristic.explain(self._position, self.node(state), player)

    def describe(self) -> str:
        """The game as the heuristics it judges with: its context and every position and move rule with its weight.
        Two facades describe alike when they judge alike, whatever else their context holds."""
        position = [] if self._position is None else json.loads(self._heuristic.describe(self._position))["position"]
        move = [] if self._move is None else json.loads(self._heuristic.describe(self._move))["move"]
        return json.dumps({"context": self._context, "position": position, "move": move}, indent=2)
