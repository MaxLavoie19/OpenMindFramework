import logging
import random

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.mcts.model.action_sample import ActionSample
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.service.predictor import Predictor
from openmind.training.constant.training_constant import SEED_RANGE
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class SelfPlay:
    """Lets an agent play a domain against itself and collects the samples of every search. Every game draws two seeds
    up front, one for its agent and one for its outcomes, so games don't depend on each other: they run in the task
    runner's workers and give the same samples whatever the number of workers."""

    def __init__(self, solver: Solver, predictor: Predictor, state_reader: StateReader, task_runner: TaskRunner) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._task_runner = task_runner

    def play(
        self, domain: Domain, agent_builder: AgentBuilder, games: int, rng: random.Random
    ) -> tuple[ActionSample, ...]:
        """The samples of every game, in game order; the builder needs its iterations and exploration set."""
        agent_seeds, outcome_seeds = [], []
        for _ in range(games):
            agent_seeds.append(rng.randrange(SEED_RANGE))
            outcome_seeds.append(rng.randrange(SEED_RANGE))
        results = self._task_runner.map(
            self.play_game, [domain] * games, [agent_builder] * games, agent_seeds, outcome_seeds
        )
        samples: list[ActionSample] = []
        for game, (game_samples, payoffs) in enumerate(results, start=1):
            samples.extend(game_samples)
            logger.info(
                "Self-play game %d: %d samples, payoffs %s",
                game,
                len(game_samples),
                " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, payoffs)),
            )
        return tuple(samples)

    def play_game(
        self, domain: Domain, agent_builder: AgentBuilder, agent_seed: int, outcome_seed: int
    ) -> tuple[tuple[ActionSample, ...], tuple[float, ...]]:
        """One game: the samples of its searches and the final payoffs."""
        agent = agent_builder.with_seed(agent_seed).build()
        rng = random.Random(outcome_seed)
        state, samples = domain.initial_state, []
        while self._solver.solve(domain.problem, state):
            result = agent.search(domain, state)
            samples.extend(result.samples)
            outcomes = self._predictor.predict(domain.transitions, state, result.chosen).outcomes
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        return tuple(samples), self._state_reader.payoffs(state, domain.players)
