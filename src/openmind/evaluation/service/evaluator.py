import logging
import math
import random
import time
from datetime import datetime

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.model.domain import Domain
from openmind.agent.model.policy import Policy
from openmind.agent.service.agent import Agent
from openmind.agent.service.random_policy import RandomPolicy
from openmind.csp.service.solver import Solver
from openmind.evaluation.constant.evaluation_constant import RANDOM_OPPONENT, UNTRAINED_OPPONENT
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.model.rater_agreement import RaterAgreement
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.mcts.model.action_rater import ActionRater
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Evaluator:
    """Measures how well an agent plays a domain: results against baselines and agreement with perfect play. When a
    rater guides the agent, agreement is also measured, on the same positions, for an unguided agent and for the rater
    alone."""

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
        rater: ActionRater | None = None,
    ) -> EvaluationReport:
        """Sets the builder's iterations and seed for every agent it builds. rules_file names what guides the agent, and
        rater is that same model, to compare the agent with an unguided one and to measure the rater alone."""
        rng = random.Random(settings.seed)
        evaluated = agent_builder.with_iterations(settings.iterations).with_seed(settings.seed).build()
        opponents: tuple[tuple[str, Policy], ...] = (
            (RANDOM_OPPONENT, RandomPolicy(self._solver, random.Random(settings.seed))),
            (UNTRAINED_OPPONENT, create_agent(settings.iterations, settings.seed)),
        )
        baselines = tuple(
            self._series(domain, evaluated, name, opponent, settings.games, rng) for name, opponent in opponents
        )
        every_action_optimal = 0
        agreement: list[Agreement] = []
        unguided_agreement: list[Agreement] = []
        rater_agreement: RaterAgreement | None = None
        if settings.positions == 0:
            logger.info("Agreement with perfect play skipped: no positions")
        else:
            positions = self._exact_search.positions(domain)
            if settings.positions is None:
                sample = list(positions)
            else:
                sample = rng.sample(positions, min(settings.positions, len(positions)))
            values = [self._exact_search.action_values(domain, state) for state in sample]
            every_action_optimal = sum(1 for action_values in values if len(self._optimal(action_values)) == len(action_values))
            logger.info("Every action is optimal in %d of %d positions", every_action_optimal, len(sample))
            for iterations in settings.budgets:
                guided = agent_builder.with_iterations(iterations).with_seed(settings.seed).build()
                agreement.append(self._agreement(domain, guided, iterations, sample, values, "Agreement"))
                if rater is not None:
                    unguided = create_agent(iterations, settings.seed)
                    unguided_agreement.append(
                        self._agreement(domain, unguided, iterations, sample, values, "Unguided agreement")
                    )
            if rater is not None:
                rater_agreement = self._rater_agreement(rater, sample, values)
        created_at = datetime.now().replace(microsecond=0)
        return EvaluationReport(
            domain.name,
            created_at,
            rules_file,
            settings,
            baselines,
            every_action_optimal,
            tuple(agreement),
            tuple(unguided_agreement),
            rater_agreement,
        )

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
        agent: Agent,
        iterations: int,
        sample: list[State],
        values: list[tuple[tuple[Action, float], ...]],
        kind: str,
    ) -> Agreement:
        optimal = 0
        shares: list[float] = []
        regrets: list[float] = []
        seconds: list[float] = []
        for state, action_values in zip(sample, values, strict=True):
            started = time.perf_counter()
            result = agent.search(domain, state)
            seconds.append(time.perf_counter() - started)
            value_of = dict(action_values)
            best = max(value_of.values())
            optimal_actions = self._optimal(action_values)
            visits = sum(item.visits for item in result.statistics)
            optimal_visits = sum(item.visits for item in result.statistics if item.action in optimal_actions)
            share = optimal_visits / visits if visits else 0.0
            regret = best - value_of[result.chosen]
            optimal += result.chosen in optimal_actions
            shares.append(share)
            regrets.append(regret)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "%s at %d iterations: chose %s, regret %s; optimal: %s; %s of visits on optimal actions; state: %s",
                    kind,
                    iterations,
                    self._action_text_mapper.to_text(result.chosen),
                    regret,
                    ", ".join(self._action_text_mapper.to_text(action) for action, _ in action_values if action in optimal_actions),
                    share,
                    self._state_text_mapper.to_text(state).replace("\n", ", "),
                )
        count = len(sample)
        agreement = Agreement(
            iterations,
            count,
            optimal,
            math.fsum(shares) / count if count else 0.0,
            math.fsum(regrets) / count if count else 0.0,
            math.fsum(seconds) / count if count else 0.0,
        )
        logger.info(
            "%s with perfect play at %d iterations: %d of %d positions, %s of visits on optimal actions, mean regret %s, "
            "%s seconds per choice",
            kind,
            iterations,
            agreement.optimal,
            agreement.positions,
            agreement.optimal_visit_share,
            agreement.mean_regret,
            agreement.seconds_per_choice,
        )
        return agreement

    def _rater_agreement(
        self, rater: ActionRater, sample: list[State], values: list[tuple[tuple[Action, float], ...]]
    ) -> RaterAgreement:
        """Picks uniformly among the top-rated actions; like the search, an action rated None gets the mean of the
        other ratings, and a position where every rating is None has every action top-rated."""
        distinguishing = 0
        optimal: list[float] = []
        regrets: list[float] = []
        for state, action_values in zip(sample, values, strict=True):
            actions = tuple(action for action, _ in action_values)
            ratings = rater.rate(state, actions)
            known = [rating for rating in ratings if rating is not None]
            fill = math.fsum(known) / len(known) if known else 0.0
            filled = [fill if rating is None else rating for rating in ratings]
            top = max(filled)
            picks = [(action, value) for (action, value), rating in zip(action_values, filled, strict=True) if math.isclose(rating, top)]
            distinguishing += len(picks) < len(actions)
            best = max(value for _, value in action_values)
            optimal_actions = self._optimal(action_values)
            optimal.append(sum(1 for action, _ in picks if action in optimal_actions) / len(picks))
            regrets.append(math.fsum(best - value for _, value in picks) / len(picks))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Rater alone: top-rated %s; optimal: %s; ratings %s; state: %s",
                    ", ".join(self._action_text_mapper.to_text(action) for action, _ in picks),
                    ", ".join(self._action_text_mapper.to_text(action) for action in actions if action in optimal_actions),
                    ", ".join(f"{self._action_text_mapper.to_text(action)}={rating}" for action, rating in zip(actions, ratings, strict=True)),
                    self._state_text_mapper.to_text(state).replace("\n", ", "),
                )
        count = len(sample)
        rater_agreement = RaterAgreement(
            count, distinguishing, math.fsum(optimal), math.fsum(regrets) / count if count else 0.0
        )
        logger.info(
            "Rater alone: ratings separate actions in %d of %d positions; a top-rated action is optimal in %s of %d; "
            "mean regret %s",
            rater_agreement.distinguishing,
            rater_agreement.positions,
            rater_agreement.optimal,
            rater_agreement.positions,
            rater_agreement.mean_regret,
        )
        return rater_agreement

    def _optimal(self, action_values: tuple[tuple[Action, float], ...]) -> frozenset[Action]:
        best = max(value for _, value in action_values)
        return frozenset(action for action, value in action_values if math.isclose(value, best))
