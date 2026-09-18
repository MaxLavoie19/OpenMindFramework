import logging
import math
import random
import warnings
from datetime import datetime
from functools import partial

import numpy as np
from scipy.stats import binomtest, wilcoxon

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION, MATCH_GAME, RANDOM_POLICY_TEXT
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.model.policy_factory import PolicyFactory
from openmind.agent.service.game_memory import GameMemory
from openmind.evaluation.constant.evaluation_constant import RANDOM_OPPONENT, REFERENCE_TOLERANCE, UNTRAINED_OPPONENT
from openmind.evaluation.factory.baseline_policy_factory import create_built_agent, create_random_policy
from openmind.evaluation.mapper.match_game_summary_mapper import MatchGameSummaryMapper
from openmind.evaluation.model.action_values import ActionValues
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.guidance_test import GuidanceTest
from openmind.evaluation.model.match_game import MatchGame
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.model.rater_agreement import RaterAgreement
from openmind.evaluation.model.value_measure import ValueMeasure
from openmind.evaluation.service.choice_measurer import ChoiceMeasurer
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.evaluation.service.reference_search import ReferenceSearch
from openmind.evaluation.service.value_measurer import ValueMeasurer
from openmind.mcts.constant.mcts_constant import UNIFORM_PRIOR
from openmind.mcts.factory.move_prior_factory import create_move_prior
from openmind.mcts.model.action_rater import ActionRater
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.time_control import TimeControl
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

type Measures = tuple[np.ndarray, np.ndarray, np.ndarray]


class Evaluator:
    """Measures how well an agent plays a rbs: results against baselines and agreement with perfect play, from exact
    search or, with reference_iterations, from long unguided searches on positions of random games. When a rater guides
    the agent or a valuer values its positions, agreement is also measured, on the same positions, for an unguided agent
    and for each model alone, and the agent is compared with the unguided one position by position with paired tests.
    Baseline games, reference searches, the positions searched at each budget and the valuer's measures run in the task
    runner's workers; every search is seeded, so the results don't depend on the number of workers."""

    def __init__(
        self,
        match_runner: MatchRunner,
        exact_search: ExactSearch,
        reference_search: ReferenceSearch,
        choice_measurer: ChoiceMeasurer,
        value_measurer: ValueMeasurer,
        task_runner: TaskRunner,
        state_text_mapper: StateTextMapper,
        action_text_mapper: ActionTextMapper,
        game_memory: GameMemory | None = None,
    ) -> None:
        self._game_memory = game_memory
        self._match_runner = match_runner
        self._exact_search = exact_search
        self._reference_search = reference_search
        self._choice_measurer = choice_measurer
        self._value_measurer = value_measurer
        self._task_runner = task_runner
        self._state_text_mapper = state_text_mapper
        self._action_text_mapper = action_text_mapper

    def evaluate(
        self,
        rbs: RuleBasedSystem,
        agent_builder: AgentBuilder,
        settings: EvaluationSettings,
        rules_file: str | None = None,
        rater: ActionRater | None = None,
        values_file: str | None = None,
        valuer: PositionValuer | None = None,
    ) -> EvaluationReport:
        """Sets the builder's iterations, seed, rollout guidance and rollout actions for every agent it builds. rules_file
        names what guides the agent and rater is that same model; values_file names what values the agent's positions and
        valuer is that same model. Either model compares the agent with an unguided one, and each is measured alone. A
        reference search samples positions, so it needs a number of positions rather than all of them. To run in several
        workers, the builder, with its models, must pickle."""
        rng = random.Random(settings.seed)
        agent_builder.with_guided_rollouts(settings.guided_rollouts).with_rollout_actions(settings.rollout_actions)
        agent_builder.with_rollout_limit(settings.rollout_limit, settings.unfinished_payoff)
        agent_builder.with_iterations(settings.iterations).with_seed(settings.seed)
        untrained = (
            AgentBuilder()
            .with_exploration(EXPLORATION)
            .with_iterations(settings.iterations)
            .with_seed(settings.seed)
            .with_rollout_limit(settings.rollout_limit, settings.unfinished_payoff)
        )
        agent_builder.with_selection(settings.selection, settings.puct_exploration).with_prior(
            create_move_prior(settings.prior, settings.prior_temperature, rbs, rater, valuer)
        )
        untrained.with_selection(settings.selection, settings.puct_exploration).with_prior(
            create_move_prior(UNIFORM_PRIOR, settings.prior_temperature, rbs)
        )
        if settings.time_control is not None:
            for builder in (agent_builder, untrained):
                builder.with_time_budget_estimator(
                    PlainTimeBudgetEstimator(settings.expected_steps, settings.time_control.base_seconds * settings.time_reserve)
                )
        opponents: tuple[tuple[str, PolicyFactory, ModelDescription], ...] = (
            (RANDOM_OPPONENT, create_random_policy, ModelDescription(RANDOM_OPPONENT, RANDOM_POLICY_TEXT)),
            (UNTRAINED_OPPONENT, partial(create_built_agent, untrained), untrained.describe(UNTRAINED_OPPONENT)),
        )
        evaluated = partial(create_built_agent, agent_builder)
        evaluated_model = agent_builder.describe(values_file or rules_file or "evaluated agent")
        baselines = tuple(
            self._series(rbs, evaluated, name, opponent, settings.games, rng, evaluated_model, model, settings.time_control)
            for name, opponent, model in opponents
        )
        every_action_optimal = 0
        agreement: list[Agreement] = []
        unguided_agreement: list[Agreement] = []
        guidance_tests: list[GuidanceTest] = []
        rater_agreement: RaterAgreement | None = None
        value_measure: ValueMeasure | None = None
        compared = rater is not None or valuer is not None
        if settings.positions == 0:
            logger.info("Agreement with perfect play skipped: no positions")
        else:
            sample, values, tolerance = self._reference(rbs, settings, rng)
            every_action_optimal = sum(
                1
                for action_values in values
                if len(self._choice_measurer.optimal(action_values, tolerance)) == len(action_values)
            )
            logger.info("Every action is optimal in %d of %d positions", every_action_optimal, len(sample))
            for iterations in settings.budgets:
                agent_builder.with_iterations(iterations).with_seed(settings.seed)
                guided_result, guided_measures = self._agreement(
                    rbs, agent_builder, iterations, sample, values, tolerance, "Agreement"
                )
                agreement.append(guided_result)
                if compared:
                    unguided = (
                        AgentBuilder()
                        .with_exploration(EXPLORATION)
                        .with_iterations(iterations)
                        .with_seed(settings.seed)
                        .with_rollout_limit(settings.rollout_limit, settings.unfinished_payoff)
                    )
                    unguided_result, unguided_measures = self._agreement(
                        rbs, unguided, iterations, sample, values, tolerance, "Unguided agreement"
                    )
                    unguided_agreement.append(unguided_result)
                    guidance_tests.append(self._guidance_test(iterations, guided_measures, unguided_measures))
            if rater is not None:
                rater_agreement = self._rater_agreement(rater, sample, values, tolerance)
            if valuer is not None:
                value_measure = self._value_measure(rbs, valuer, sample, values, tolerance)
        created_at = datetime.now().replace(microsecond=0)
        return EvaluationReport(
            rbs.context,
            created_at,
            rules_file,
            settings,
            baselines,
            every_action_optimal,
            tuple(agreement),
            tuple(unguided_agreement),
            rater_agreement,
            tuple(guidance_tests),
            values_file,
            value_measure,
        )

    def _reference(
        self, rbs: RuleBasedSystem, settings: EvaluationSettings, rng: random.Random
    ) -> tuple[list[State], list[ActionValues], float]:
        """The positions to measure, every legal action's value in each, and how far from the best an optimal action's
        value may be."""
        if settings.reference_iterations is None:
            positions = self._exact_search.positions(rbs)
            if settings.positions is None:
                sample = list(positions)
            else:
                sample = rng.sample(positions, min(settings.positions, len(positions)))
            return sample, [self._exact_search.action_values(rbs, state) for state in sample], 0.0
        if settings.positions is None:
            raise ValueError("A reference search samples positions: give a number of positions, not all of them")
        sample = list(self._reference_search.positions(rbs, settings.positions, rng))
        count = len(sample)
        values = self._task_runner.map(
            self._reference_search.action_values,
            [rbs] * count,
            sample,
            [settings.reference_iterations] * count,
            [settings.seed] * count,
        )
        logger.info(
            "Reference values from %d-iteration unguided searches on %d positions",
            settings.reference_iterations,
            len(sample),
        )
        return sample, values, REFERENCE_TOLERANCE

    def _series(
        self,
        rbs: RuleBasedSystem,
        evaluated: PolicyFactory,
        name: str,
        opponent: PolicyFactory,
        games: int,
        rng: random.Random,
        evaluated_model: ModelDescription,
        opponent_model: ModelDescription,
        time_control: TimeControl | None = None,
    ) -> MatchResults:
        memory = self._game_memory
        on_game = None
        if memory is not None:
            mapper = MatchGameSummaryMapper()

            def remember(index: int, seat: int, game: MatchGame) -> None:
                record = rbs.record(game.actions, game.flagged, game.payoffs) if game.actions else None
                memory.remember(
                    mapper.to_summary(
                        rbs, game, MATCH_GAME, None, index + 1, seat, evaluated_model, opponent_model, record, time_control
                    )
                )

            on_game = remember
        results = self._match_runner.series(rbs, evaluated, opponent, name, games, rng, time_control, on_game)
        logger.info(
            "Against %s: %d games, %d wins, %d draws, %d losses%s",
            name,
            results.games,
            results.wins,
            results.draws,
            results.losses,
            _on_time(results),
        )
        return results

    def _agreement(
        self,
        rbs: RuleBasedSystem,
        agent_builder: AgentBuilder,
        iterations: int,
        sample: list[State],
        values: list[ActionValues],
        tolerance: float,
        kind: str,
    ) -> tuple[Agreement, Measures]:
        """The agreement, and per position whether the choice was optimal, the share of visits on low-value actions and
        the regret. The positions are split between the workers, each slice searched by an agent built once."""
        slices = self._task_runner.split(list(zip(sample, values, strict=True)))
        count = len(slices)
        results = self._task_runner.map(
            self._choice_measurer.measure,
            [rbs] * count,
            [agent_builder] * count,
            slices,
            [tolerance] * count,
            [kind] * count,
            [iterations] * count,
        )
        measures = [measure for result in results for measure in result]
        positions = len(measures)
        agreement = Agreement(
            iterations,
            positions,
            sum(measure.optimal for measure in measures),
            math.fsum(measure.optimal_visit_share for measure in measures) / positions if positions else 0.0,
            math.fsum(measure.regret for measure in measures) / positions if positions else 0.0,
            math.fsum(measure.seconds for measure in measures) / positions if positions else 0.0,
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
        return agreement, (
            np.array([measure.optimal for measure in measures], dtype=bool),
            1.0 - np.array([measure.optimal_visit_share for measure in measures], dtype=float),
            np.array([measure.regret for measure in measures], dtype=float),
        )

    def _guidance_test(self, iterations: int, guided: Measures, unguided: Measures) -> GuidanceTest:
        guided_optimal, guided_low_value, guided_regret = guided
        unguided_optimal, unguided_low_value, unguided_regret = unguided
        only_guided = int(np.count_nonzero(guided_optimal & ~unguided_optimal))
        only_unguided = int(np.count_nonzero(unguided_optimal & ~guided_optimal))
        discordant = only_guided + only_unguided
        low_value_differences = guided_low_value - unguided_low_value
        regret_differences = guided_regret - unguided_regret
        test = GuidanceTest(
            iterations,
            len(low_value_differences),
            float(low_value_differences.mean()) if len(low_value_differences) else 0.0,
            self._signed_rank(low_value_differences),
            float(regret_differences.mean()) if len(regret_differences) else 0.0,
            self._signed_rank(regret_differences),
            only_guided,
            only_unguided,
            float(binomtest(only_guided, discordant, 0.5).pvalue) if discordant else 1.0,
        )
        logger.info(
            "Guided against unguided at %d iterations on %d positions: low-value visit share %+f (p %s), regret %+f "
            "(p %s), optimal choice only guided %d, only unguided %d (p %s)",
            test.iterations,
            test.positions,
            test.low_value_share_difference,
            test.low_value_share_p,
            test.regret_difference,
            test.regret_p,
            test.optimal_only_guided,
            test.optimal_only_unguided,
            test.optimal_choice_p,
        )
        return test

    def _signed_rank(self, differences: np.ndarray) -> float:
        """The two-sided Wilcoxon signed-rank p-value, zero differences split between the signs; 1 without a nonzero
        difference."""
        if not np.any(differences != 0):
            return 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(wilcoxon(differences, zero_method="zsplit").pvalue)

    def _rater_agreement(
        self, rater: ActionRater, sample: list[State], values: list[ActionValues], tolerance: float
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
            optimal_actions = self._choice_measurer.optimal(action_values, tolerance)
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

    def _value_measure(
        self,
        rbs: RuleBasedSystem,
        valuer: PositionValuer,
        sample: list[State],
        values: list[ActionValues],
        tolerance: float,
    ) -> ValueMeasure:
        """The positions are split between the workers, each slice measured with the same valuer."""
        slices = self._task_runner.split(list(zip(sample, values, strict=True)))
        count = len(slices)
        results = self._task_runner.map(
            self._value_measurer.measure, [rbs] * count, [valuer] * count, slices, [tolerance] * count
        )
        measures = [measure for result in results for measure in result]
        errors = [error for error, _, _ in measures if error is not None]
        positions = len(measures)
        value_measure = ValueMeasure(
            positions,
            len(errors),
            math.fsum(errors) / len(errors) if errors else None,
            math.fsum(optimal for _, optimal, _ in measures),
            math.fsum(regret for _, _, regret in measures) / positions if positions else 0.0,
        )
        logger.info(
            "Values alone: valued %d of %d positions, mean absolute error %s against the best action's value; one step "
            "ahead, a top-valued action is optimal in %s of %d; mean regret %s",
            value_measure.valued,
            value_measure.positions,
            value_measure.mean_absolute_error,
            value_measure.optimal,
            value_measure.positions,
            value_measure.mean_regret,
        )
        return value_measure


def _on_time(results: MatchResults) -> str:
    """`, on <time control>: <n> wins and <m> losses on time` for a series played on a clock; nothing otherwise."""
    if results.time_control is None:
        return ""
    control = TimeControlTextMapper().to_text(results.time_control)
    return f", on {control}: {results.wins_on_time} wins and {results.losses_on_time} losses on time"
