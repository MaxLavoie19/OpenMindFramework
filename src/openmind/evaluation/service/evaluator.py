import logging
import random
import time
from datetime import datetime

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.model.domain import Domain
from openmind.agent.model.policy import Policy
from openmind.agent.service.random_policy import RandomPolicy
from openmind.csp.service.solver import Solver
from openmind.evaluation.constant.evaluation_constant import RANDOM_OPPONENT, UNTRAINED_OPPONENT
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Evaluator:
    """Measures how well an agent plays a domain: results against baselines and agreement with perfect play."""

    def __init__(
        self,
        match_runner: MatchRunner,
        exact_search: ExactSearch,
        solver: Solver,
        state_text_mapper: StateTextMapper,
        action_text_mapper: ActionTextMapper,
    ) -> None:
        self._match_runner = match_runner
        self._exact_search = exact_search
        self._solver = solver
        self._state_text_mapper = state_text_mapper
        self._action_text_mapper = action_text_mapper

    def evaluate(
        self,
        domain: Domain,
        agent_builder: AgentBuilder,
        settings: EvaluationSettings,
        rules_file: str | None = None,
    ) -> EvaluationReport:
        """Sets the builder's iterations and seed for every agent it builds; rules_file names what guides the agent."""
        rng = random.Random(settings.seed)
        evaluated = agent_builder.with_iterations(settings.iterations).with_seed(settings.seed).build()
        opponents: tuple[tuple[str, Policy], ...] = (
            (RANDOM_OPPONENT, RandomPolicy(self._solver, random.Random(settings.seed))),
            (UNTRAINED_OPPONENT, create_agent(settings.iterations, settings.seed)),
        )
        baselines = tuple(
            self._series(domain, evaluated, name, opponent, settings.games, rng) for name, opponent in opponents
        )
        agreement: tuple[Agreement, ...] = ()
        if settings.positions == 0:
            logger.info("Agreement with perfect play skipped: no positions")
        else:
            positions = self._exact_search.positions(domain)
            sample = rng.sample(positions, min(settings.positions, len(positions)))
            agreement = tuple(
                self._agreement(domain, agent_builder, iterations, settings.seed, sample)
                for iterations in settings.budgets
            )
        created_at = datetime.now().replace(microsecond=0)
        return EvaluationReport(domain.name, created_at, rules_file, settings, baselines, agreement)

    def _series(
        self,
        domain: Domain,
        evaluated: Policy,
        name: str,
        opponent: Policy,
        games: int,
        rng: random.Random,
    ) -> MatchResults:
        results = self._match_runner.series(domain, evaluated, opponent, name, games, rng)
        logger.info(
            "Against %s: %d games, %d wins, %d draws, %d losses",
            name,
            results.games,
            results.wins,
            results.draws,
            results.losses,
        )
        return results

    def _agreement(
        self,
        domain: Domain,
        agent_builder: AgentBuilder,
        iterations: int,
        seed: int,
        sample: list[State],
    ) -> Agreement:
        agent = agent_builder.with_iterations(iterations).with_seed(seed).build()
        optimal = 0
        seconds = 0.0
        for state in sample:
            started = time.perf_counter()
            chosen = agent.choose(domain, state)
            seconds += time.perf_counter() - started
            best = self._exact_search.optimal_actions(domain, state)
            if chosen in best:
                optimal += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "At %d iterations, chose %s; optimal: %s; state: %s",
                    iterations,
                    self._action_text_mapper.to_text(chosen),
                    ", ".join(self._action_text_mapper.to_text(action) for action in best),
                    self._state_text_mapper.to_text(state).replace("\n", ", "),
                )
        seconds_per_choice = seconds / len(sample) if sample else 0.0
        logger.info(
            "Agreement with perfect play at %d iterations: %d of %d positions, %s seconds per choice",
            iterations,
            optimal,
            len(sample),
            seconds_per_choice,
        )
        return Agreement(iterations, len(sample), optimal, seconds_per_choice)
