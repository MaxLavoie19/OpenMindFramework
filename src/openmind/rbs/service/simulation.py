import logging

from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.service.solver import Solver
from openmind.knowledge.constant.rule_kind_constant import ENDING, INITIAL, PLAYERS
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.predictor.service.rule_predictor import RulePredictor
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.constant.rule_constant import EFFECTS_DEFINITIONS, RULES_DEFINITIONS
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.players import Players
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: The state a rule that reads nothing is called with.
NOTHING = State(())


class Simulation:
    """Runs a simulation ruleset's RBS: where the game starts, who plays, the players acting, their legal actions
    solved by the CSP, and what actions taken at once lead to, through the rule predictor. It keeps nothing: built once,
    it is given the RBS with every call.

    All players play at the same time, all the time: the players acting in a state are those with a legal action
    there, the constraints reading each as `player`, and a game played in turns leaves a player no action outside
    their turn."""

    def __init__(self, solver: Solver, predictor: RulePredictor, rule_caller: RuleCaller) -> None:
        self._solver = solver
        self._predictor = predictor
        self._rule_caller = rule_caller

    def start(self, rbs: RuleBasedSystem) -> State:
        """Where the game starts."""
        read = self._read(rbs, INITIAL, "where the game starts")
        return read if isinstance(read, State) else State(tuple(read))  # type: ignore[arg-type]

    def players(self, rbs: RuleBasedSystem) -> Players:
        """Who plays, and the Map holding each player's payoff."""
        read = self._read(rbs, PLAYERS, "who plays")
        if isinstance(read, Players):
            return read
        names, payoff = read  # type: ignore[misc]
        return Players(tuple(names), payoff)

    def actions_with_statistics(
        self, rbs: RuleBasedSystem, state: State, limit: int | None = None, player: str | None = None
    ) -> tuple[tuple[Action, ...], SolveStatistics]:
        """The player's legal actions, at most limit of them, with what the search did, summed over the game's
        actions."""
        found: list[Action] = []
        assignments = dead_ends = pruned_values = 0
        definitions = rbs.definitions(RULES_DEFINITIONS)
        for name in rbs.action_names():
            remaining = None if limit is None else limit - len(found)
            if remaining == 0:
                break
            solved, statistics = self._solver.solve_with_statistics(
                state, name, rbs.values(name), rbs.constraints(name), definitions, remaining, player
            )
            found.extend(solved)
            assignments += statistics.assignments
            dead_ends += statistics.dead_ends
            pruned_values += statistics.pruned_values
        return tuple(found), SolveStatistics(len(found), assignments, dead_ends, pruned_values)

    def actions(
        self, rbs: RuleBasedSystem, state: State, limit: int | None = None, player: str | None = None
    ) -> tuple[Action, ...]:
        """The player's legal actions in the state, at most limit of them. Without a player, those of the one player
        acting: none once no player has an action, and several players acting at once raise ValueError, their actions
        being joint."""
        if player is not None:
            return self.actions_with_statistics(rbs, state, limit, player)[0]
        legal = [(name, self.actions_with_statistics(rbs, state, limit, name)[0]) for name in self.players(rbs).names]
        acting = [(name, actions) for name, actions in legal if actions]
        if len(acting) > 1:
            raise ValueError(f"{', '.join(name for name, _ in acting)} act at once: their actions are joint (see joint_actions)")
        return acting[0][1] if acting else ()

    def acting(self, rbs: RuleBasedSystem, state: State) -> tuple[int, ...]:
        """The players acting in the state, by index in the players' names: those with at least one legal action."""
        names = self.players(rbs).names
        return tuple(index for index, name in enumerate(names) if self.actions(rbs, state, limit=1, player=name))

    def acting_player(self, rbs: RuleBasedSystem, state: State) -> str:
        """The one player acting in the state; no player, or several acting at once, raise ValueError."""
        names = self.players(rbs).names
        acting = self.acting(rbs, state)
        if len(acting) != 1:
            raise ValueError(f"{', '.join(names[index] for index in acting) or 'No player'} can act, not exactly one player")
        return names[acting[0]]

    def joint_actions(self, rbs: RuleBasedSystem, state: State) -> tuple[tuple[int, tuple[Action, ...]], ...]:
        """Each player with a legal action, by index in the players' names, with its legal actions: what the players
        acting at once choose from. Empty once no player has an action."""
        legal = ((index, self.actions(rbs, state, player=name)) for index, name in enumerate(self.players(rbs).names))
        return tuple((index, actions) for index, actions in legal if actions)

    def outcomes(self, rbs: RuleBasedSystem, state: State, action: Action) -> OutcomeDistribution:
        """What one action leads to, with each outcome's chance, run without a player."""
        return self._predictor.predict_action(rbs, state, action)

    def joint_outcomes(self, rbs: RuleBasedSystem, state: State, joint: JointAction) -> OutcomeDistribution:
        """What the players' actions, taken at once, lead to together."""
        return self._predictor.predict(rbs, state, joint)

    def ended(self, rbs: RuleBasedSystem, state: State) -> str | None:
        """Why the game ended, or None while it goes on and where the game doesn't say."""
        value = self.call(rbs, ENDING, state)
        return None if value is None else str(value)

    def call(self, rbs: RuleBasedSystem, kind: str, state: State, parameters: dict[str, object] | None = None) -> object:
        """What the RBS's rule of that kind says about the state, or None where it has no such rule. A rule that raises
        gives None with a warning: it is the project's code, and the logs must survive it."""
        rules = rbs.of(kind)
        if not rules:
            return None
        try:
            return self._rule_caller.value(rules[0].rule, state, parameters, None, rbs.definitions(EFFECTS_DEFINITIONS))
        except Exception:  # noqa: BLE001 - a game's rule is the project's code
            logger.warning("The %s %s rule raised", rbs.context, kind, exc_info=True)
            return None

    def _read(self, rbs: RuleBasedSystem, kind: str, what: str) -> object:
        """What a rule that reads nothing gives; an RBS without it isn't a game."""
        rules = rbs.of(kind)
        if not rules:
            raise ValueError(f"{rbs.context} has no rule saying {what}")
        return self._rule_caller.value(rules[0].rule, NOTHING)
