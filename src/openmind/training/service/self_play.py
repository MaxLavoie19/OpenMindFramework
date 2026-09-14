import logging
import math
import random

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.service.predictor import Predictor
from openmind.training.constant.training_constant import SEED_RANGE
from openmind.training.model.played_game import PlayedGame
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class SelfPlay:
    """Lets an agent play a domain against itself and keeps each game's searches, positions and payoffs. Every game draws
    two seeds up front, one for its agent and one for its outcomes, so games don't depend on each other: they run in the
    task runner's workers and give the same games whatever the number of workers."""

    def __init__(self, solver: Solver, predictor: Predictor, state_reader: StateReader, task_runner: TaskRunner) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._task_runner = task_runner

    def play(self, domain: Domain, agent_builder: AgentBuilder, games: int, rng: random.Random) -> tuple[PlayedGame, ...]:
        """Every game, in order; the builder needs its iterations and exploration set."""
        agent_seeds, outcome_seeds = [], []
        for _ in range(games):
            agent_seeds.append(rng.randrange(SEED_RANGE))
            outcome_seeds.append(rng.randrange(SEED_RANGE))
        played = self._task_runner.map(
            self.play_game, [domain] * games, [agent_builder] * games, agent_seeds, outcome_seeds
        )
        for game, result in enumerate(played, start=1):
            logger.info(
                "Self-play game %d: %d samples, payoffs %s",
                game,
                len(result.samples),
                " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, result.payoffs)),
            )
        return tuple(played)

    def play_game(self, domain: Domain, agent_builder: AgentBuilder, agent_seed: int, outcome_seed: int) -> PlayedGame:
        """One game: the samples of its searches, its positions with the search's mean payoff in each, and the final
        payoffs."""
        agent = agent_builder.with_seed(agent_seed).build()
        rng = random.Random(outcome_seed)
        state, samples, states, search_values = domain.initial_state, [], [], []
        while self._solver.solve(domain.problem, state):
            result = agent.search(domain, state)
            samples.extend(result.samples)
            visits = sum(item.visits for item in result.statistics)
            states.append(state)
            search_values.append(math.fsum(item.visits * item.mean_payoff for item in result.statistics) / visits)
            outcomes = self._predictor.predict(domain.transitions, state, result.chosen).outcomes
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        return PlayedGame(tuple(samples), tuple(states), tuple(search_values), self._state_reader.payoffs(state, domain.players))
