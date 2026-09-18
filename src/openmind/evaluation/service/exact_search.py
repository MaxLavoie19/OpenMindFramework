import logging
import math

from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class ExactSearch:
    """Perfect play by searching every reachable state; each state's value is computed once per game."""

    def __init__(self, state_reader: StateReader) -> None:
        self._state_reader = state_reader
        self._values: dict[tuple[str, State], tuple[float, ...]] = {}

    def __getstate__(self) -> dict[str, object]:
        """The values stay behind when the search is copied to another process, which computes its own."""
        return {name: value for name, value in self.__dict__.items() if name != "_values"}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._values = {}

    def action_values(self, rbs: RuleBasedSystem, state: State) -> tuple[tuple[Action, float], ...]:
        """Each legal action with its expected payoff for the player to act under perfect play, in the solver's order."""
        actions = rbs.actions(state)
        if not actions:
            raise ValueError("No legal action in this state")
        player = self._state_reader.player_to_act(state, rbs.players())
        return tuple((action, self._action_value(rbs, state, action)[player]) for action in actions)

    def optimal_actions(self, rbs: RuleBasedSystem, state: State) -> tuple[Action, ...]:
        """The legal actions with the highest expected payoff for the player to act, in the solver's order."""
        values = self.action_values(rbs, state)
        best = max(value for _, value in values)
        return tuple(action for action, value in values if math.isclose(value, best))

    def value(self, rbs: RuleBasedSystem, state: State) -> tuple[float, ...]:
        """Each player's expected payoff from the state under perfect play, in the order of the players' names."""
        return self._value(rbs, state)

    def positions(self, rbs: RuleBasedSystem) -> tuple[State, ...]:
        """Every state reachable from the initial state that has a legal action."""
        seen: set[State] = set()
        positions: list[State] = []
        pending = [rbs.start()]
        while pending:
            state = pending.pop()
            if state in seen:
                continue
            seen.add(state)
            actions = rbs.actions(state)
            if actions:
                positions.append(state)
            for action in actions:
                outcomes = rbs.outcomes(state, action).outcomes
                pending.extend(outcome for outcome, _ in outcomes)
        logger.info("%s has %d positions with a legal action", rbs.context, len(positions))
        return tuple(positions)

    def _value(self, rbs: RuleBasedSystem, state: State) -> tuple[float, ...]:
        key = (rbs.context, state)
        if key not in self._values:
            actions = rbs.actions(state)
            if actions:
                player = self._state_reader.player_to_act(state, rbs.players())
                self._values[key] = max(
                    (self._action_value(rbs, state, action) for action in actions),
                    key=lambda value: value[player],
                )
            else:
                self._values[key] = self._state_reader.payoffs(state, rbs.players())
        return self._values[key]

    def _action_value(self, rbs: RuleBasedSystem, state: State, action: Action) -> tuple[float, ...]:
        outcomes = [
            (self._value(rbs, outcome), probability)
            for outcome, probability in rbs.outcomes(state, action).outcomes
        ]
        return tuple(
            math.fsum(probability * value[index] for value, probability in outcomes)
            for index in range(len(rbs.players().names))
        )
