import logging
import math
import random

from openmind.csp.model.problem import Problem
from openmind.csp.service.solver import Solver
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.chance_node import ChanceNode
from openmind.mcts.model.decision_node import DecisionNode
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class TreeSearch:
    """Monte-Carlo Tree Search: UCT selection, chance nodes for outcomes, random rollouts."""

    def __init__(
        self,
        solver: Solver,
        predictor: Predictor,
        state_reader: StateReader,
        action_text_mapper: ActionTextMapper,
    ) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._action_text_mapper = action_text_mapper

    def search(
        self,
        problem: Problem,
        transitions: TransitionModel,
        players: Players,
        state: State,
        settings: SearchSettings,
    ) -> SearchResult:
        rng = random.Random(settings.seed)
        root = self._decision_node(problem, players, state, rng)
        if root.player is None:
            raise ValueError("No legal action to search from")
        player = players.names[root.player]
        logger.info("Searching %d iterations for %s", settings.iterations, player)
        for iteration in range(1, settings.iterations + 1):
            self._iterate(iteration, root, problem, transitions, players, settings.exploration, rng)
        statistics = tuple(self._statistics(root, root.player, action) for action in root.actions)
        chosen = max(statistics, key=lambda item: item.visits).action
        for item in statistics:
            logger.info(
                "%s: %d visits, mean payoff %s for %s",
                self._action_text_mapper.to_text(item.action),
                item.visits,
                item.mean_payoff,
                player,
            )
        logger.info("Most visited: %s", self._action_text_mapper.to_text(chosen))
        return SearchResult(player, statistics, chosen)

    def _iterate(
        self,
        iteration: int,
        root: DecisionNode,
        problem: Problem,
        transitions: TransitionModel,
        players: Players,
        exploration: float,
        rng: random.Random,
    ) -> None:
        node = root
        decisions = [root]
        chances: list[ChanceNode] = []
        while node.player is not None:
            if node.untried:
                action = node.untried.pop()
                outcomes = self._predictor.predict(transitions, node.state, action).outcomes
                chance = ChanceNode(action, outcomes, {}, 0, [0.0] * len(players.names))
                node.children[action] = chance
            else:
                chance = node.children[self._select(node, node.player, exploration)]
            chances.append(chance)
            node = self._outcome(chance, problem, players, rng)
            decisions.append(node)
            if node.visits == 0:
                break

        state, rollout_length = node.state, 0
        if node.player is not None:
            state, rollout_length = self._rollout(problem, transitions, state, rng)
        payoffs = self._payoffs(players, state)

        for decision in decisions:
            decision.visits += 1
        for chance in chances:
            chance.visits += 1
            for index, payoff in enumerate(payoffs):
                chance.payoff_sums[index] += payoff

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Iteration %d: %s, rollout of %d actions, payoffs %s",
                iteration,
                " > ".join(self._action_text_mapper.to_text(chance.action) for chance in chances),
                rollout_length,
                " ".join(f"{name}={payoff}" for name, payoff in zip(players.names, payoffs)),
            )

    def _select(self, node: DecisionNode, player: int, exploration: float) -> Action:
        log_visits = math.log(node.visits)

        def uct(action: Action) -> float:
            chance = node.children[action]
            mean = chance.payoff_sums[player] / chance.visits
            return mean + exploration * math.sqrt(log_visits / chance.visits)

        return max(node.actions, key=uct)

    def _outcome(
        self, chance: ChanceNode, problem: Problem, players: Players, rng: random.Random
    ) -> DecisionNode:
        state = self._draw(chance.outcomes, rng)
        child = chance.children.get(state)
        if child is None:
            child = self._decision_node(problem, players, state, rng)
            chance.children[state] = child
        return child

    def _decision_node(
        self, problem: Problem, players: Players, state: State, rng: random.Random
    ) -> DecisionNode:
        actions = self._solver.solve(problem, state)
        untried = list(actions)
        rng.shuffle(untried)
        player = players.names.index(str(self._state_reader.value(state, players.to_act))) if actions else None
        return DecisionNode(state, actions, untried, {}, 0, player)

    def _rollout(
        self, problem: Problem, transitions: TransitionModel, state: State, rng: random.Random
    ) -> tuple[State, int]:
        length = 0
        while actions := self._solver.solve(problem, state):
            outcomes = self._predictor.predict(transitions, state, rng.choice(actions)).outcomes
            state = self._draw(outcomes, rng)
            length += 1
        return state, length

    def _draw(self, outcomes: tuple[tuple[State, float], ...], rng: random.Random) -> State:
        (state,) = rng.choices(
            [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
        )
        return state

    def _payoffs(self, players: Players, state: State) -> tuple[float, ...]:
        payoffs: list[float] = []
        for name in players.payoffs:
            value = self._state_reader.value(state, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"No legal action is left but {name} is {value!r}, not a number")
            payoffs.append(float(value))
        return tuple(payoffs)

    def _statistics(self, root: DecisionNode, player: int, action: Action) -> ActionStatistics:
        chance = root.children.get(action)
        if chance is None or chance.visits == 0:
            return ActionStatistics(action, 0, 0.0)
        return ActionStatistics(action, chance.visits, chance.payoff_sums[player] / chance.visits)
