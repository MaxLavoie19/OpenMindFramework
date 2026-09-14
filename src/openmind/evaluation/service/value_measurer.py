import math
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.evaluation.model.action_values import ActionValues
from openmind.evaluation.service.choice_measurer import ChoiceMeasurer
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.predictor.service.predictor import Predictor
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

type PositionMeasure = tuple[float | None, float, float]


class ValueMeasurer:
    """Measures a valuer alone against the values of every legal action: how far its value of a position is from the best
    action's value, and how it chooses one step ahead. An action is worth its outcomes' values for the player to act,
    weighted by their probabilities: a finished game's payoffs, or the valuer's value of a position in play. Like a
    search with ratings, an action the valuer knows nothing about gets the mean of the others' values, and when it knows
    nothing about any, every action is top-valued."""

    def __init__(
        self, solver: Solver, predictor: Predictor, state_reader: StateReader, choice_measurer: ChoiceMeasurer
    ) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._choice_measurer = choice_measurer

    def measure(
        self,
        domain: Domain,
        valuer: PositionValuer,
        positions: Sequence[tuple[State, ActionValues]],
        tolerance: float,
    ) -> tuple[PositionMeasure, ...]:
        """Per position: the absolute error of the valuer's value for the player to act (None when it knows nothing about
        the position), the share of the top-valued actions that are optimal, and their mean regret."""
        measures: list[PositionMeasure] = []
        for state, action_values in positions:
            player = self._state_reader.player_to_act(state, domain.players)
            best = max(value for _, value in action_values)
            values = valuer.value(state)
            estimates = [self._action_value(domain, valuer, state, action, player) for action, _ in action_values]
            known = [estimate for estimate in estimates if estimate is not None]
            fill = math.fsum(known) / len(known) if known else 0.0
            filled = [fill if estimate is None else estimate for estimate in estimates]
            top = max(filled)
            picks = [
                (action, value)
                for (action, value), estimate in zip(action_values, filled, strict=True)
                if math.isclose(estimate, top)
            ]
            optimal_actions = self._choice_measurer.optimal(action_values, tolerance)
            measures.append(
                (
                    None if values is None else abs(values[player] - best),
                    sum(1 for action, _ in picks if action in optimal_actions) / len(picks),
                    math.fsum(best - value for _, value in picks) / len(picks),
                )
            )
        return tuple(measures)

    def _action_value(
        self, domain: Domain, valuer: PositionValuer, state: State, action: Action, player: int
    ) -> float | None:
        total = 0.0
        for outcome, probability in self._predictor.predict(domain.transitions, state, action).outcomes:
            if self._solver.solve(domain.problem, outcome):
                values = valuer.value(outcome)
                if values is None:
                    return None
                total += probability * values[player]
            else:
                total += probability * self._state_reader.payoffs(outcome, domain.players)[player]
        return total
