import logging
import random

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.mcts.model.action_sample import ActionSample
from openmind.predictor.service.predictor import Predictor
from openmind.training.constant.training_constant import SEED_RANGE
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class SelfPlay:
    """Lets an agent play a domain against itself and collects the samples of every search."""

    def __init__(self, solver: Solver, predictor: Predictor, state_reader: StateReader) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader

    def play(
        self, domain: Domain, agent_builder: AgentBuilder, games: int, rng: random.Random
    ) -> tuple[ActionSample, ...]:
        """Builds a differently seeded agent for every game; the builder needs its iterations and exploration set."""
        samples: list[ActionSample] = []
        for game in range(1, games + 1):
            agent = agent_builder.with_seed(rng.randrange(SEED_RANGE)).build()
            state, game_samples = domain.initial_state, 0
            while self._solver.solve(domain.problem, state):
                result = agent.search(domain, state)
                samples.extend(result.samples)
                game_samples += len(result.samples)
                outcomes = self._predictor.predict(domain.transitions, state, result.chosen).outcomes
                (state,) = rng.choices(
                    [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
                )
            payoffs = self._state_reader.payoffs(state, domain.players)
            logger.info(
                "Self-play game %d: %d samples, payoffs %s",
                game,
                game_samples,
                " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, payoffs)),
            )
        return tuple(samples)
