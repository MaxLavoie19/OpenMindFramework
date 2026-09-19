import json
import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.debug.factory.debugger_factory import process_debugger
from openmind.knowledge.constant.rule_kind_constant import MOVE, PICTURE, POSITION, RECORD
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.rbs.constant.game_record_constant import ACTIONS, PAYOFFS
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rbs.service.simulation import Simulation
from openmind.rule.constant.rule_constant import RULES_DEFINITIONS
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.players import Players
from openmind.world.model.state import State

if TYPE_CHECKING:
    from openmind.rbs.service.consequence_library import ConsequenceLibrary

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
        rule_caller: RuleCaller,
        consequence_library: "ConsequenceLibrary | None" = None,
        context_id: str | None = None,
    ) -> None:
        self._context = context
        self._context_id = context if context_id is None else context_id
        self._game = simulation_rbs
        self._heuristics = heuristics
        self._simulation = simulation
        self._rule_caller = rule_caller
        self._consequence_library = consequence_library
        systems = (() if simulation_rbs is None else (simulation_rbs,)) + heuristics
        self._rules = tuple(rule for system in systems for rule, _ in system.rules)
        self._weighted = {
            kind: tuple((rule, weight) for system in heuristics for rule, weight in system.rules if rule.kind == kind)
            for kind in (POSITION, MOVE)
        }
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
        for kind in (POSITION, MOVE):
            for held, weight in self._weighted[kind]:
                if held.id == rule.id:
                    return weight
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

    def value(self, state: State, player: str) -> float | None:
        """What the position is worth to the player, by the position heuristics: each rule's reading times its weight,
        summed. None where the context has no such rule, or where none could be read."""
        rules = self._weighted[POSITION]
        if not rules:
            return None
        with process_debugger().frame("evaluation", context=self._context, state=state, details={"player": player}):
            return self._weighed(rules, state, self._names(state, player))

    def values(self, state: State) -> tuple[float, ...] | None:
        """Each player's value, in the order of the players' names; None where any of them can't be valued."""
        valued = [self.value(state, player) for player in self.players().names]
        return None if any(value is None for value in valued) else tuple(valued)  # type: ignore[arg-type]

    def rate(
        self, state: State, actions: tuple[Action, ...], player: str | None = None
    ) -> tuple[float | None, ...]:
        """What each move is worth to the player taking it, by the move heuristics: each rule's reading times its
        weight, summed. Without a player, the one player acting in the state. None for a move where there is no such
        rule, or where none could be read."""
        rules = self._weighted[MOVE]
        if not rules:
            return (None,) * len(actions)
        player = self.acting_player(state) if player is None else player
        names = self._names(state, player)
        with process_debugger().frame("evaluation", context=self._context, state=state, details={"moves": len(actions)}):
            return tuple(
                self._weighed(rules, state, names | {"action": action.name} | dict(action.parameters)) for action in actions
            )

    def describe(self) -> str:
        """The RBS as the heuristics it judges with: its context and every position and move rule with its weight
        there. Two RBSs describe alike when they judge alike, whatever else their context holds."""
        return json.dumps(
            {
                "context": self._context,
                "position": [[rule.name, weight] for rule, weight in self._weighted[POSITION]],
                "move": [[rule.name, weight] for rule, weight in self._weighted[MOVE]],
            },
            indent=2,
        )

    def explain(self, state: State, player: str) -> tuple[tuple[RuleRecord, float], ...]:
        """Each position heuristic with what it adds to the player's value: its weight here times its reading."""
        return self._readings(self._weighted[POSITION], state, self._names(state, player))

    def _names(self, state: State, player: str) -> dict[str, object]:
        """What a heuristic reads besides the state's variables: `me`, `other`, `win_chance`, `wins`, `near`, `here`."""
        if self._consequence_library is None:
            return {"me": player}
        return self._consequence_library.names(self, state, player)

    def _weighed(
        self, rules: Sequence[tuple[RuleRecord, float]], state: State, names: dict[str, object]
    ) -> float | None:
        """The rules' readings, each times its weight in this context, summed; None where none could be read."""
        readings = self._readings(rules, state, names)
        return math.fsum(added for _, added in readings) if readings else None

    def _readings(
        self, rules: Sequence[tuple[RuleRecord, float]], state: State, names: dict[str, object]
    ) -> tuple[tuple[RuleRecord, float], ...]:
        """Each rule with what it adds: its weight here times its reading. A rule reading nothing at this moment adds
        nothing; one that raises, or gives something other than a finite number, is left out."""
        added: list[tuple[RuleRecord, float]] = []
        definitions = self._definitions()
        for rule, weight in rules:
            try:
                read = self._rule_caller.value(rule.rule, state, None, names, definitions)  # type: ignore[arg-type]
            except (KeyError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
                continue
            if read is None:
                added.append((rule, 0.0))
                continue
            if not isinstance(read, bool | int | float | np.bool_ | np.number):
                continue
            reading = float(read)  # type: ignore[arg-type]
            if math.isfinite(reading):
                added.append((rule, weight * reading))
        return tuple(added)

    def _definitions(self) -> object:
        return None if self._game is None else self._game.definitions(RULES_DEFINITIONS)
