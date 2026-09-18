import logging
import random
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from functools import partial

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION, MATCH_GAME, RANDOM_POLICY_TEXT
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.model.policy_factory import PolicyFactory
from openmind.agent.service.game_memory import GameMemory
from openmind.evaluation.constant.evaluation_constant import RANDOM_OPPONENT, UNTRAINED_OPPONENT
from openmind.evaluation.factory.baseline_policy_factory import create_random_policy, create_seeded_agent
from openmind.evaluation.mapper.match_game_summary_mapper import MatchGameSummaryMapper
from openmind.evaluation.model.match_game import MatchGame
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.mcts.constant.mcts_constant import UNIFORM_PRIOR, VALUE_PRIOR
from openmind.mcts.factory.move_prior_factory import create_move_prior
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.model.time_control import TimeControl
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.training.constant.training_constant import START_RULES
from openmind.training.model.training_report import TrainingReport
from openmind.training.model.training_round import TrainingRound
from openmind.training.model.value_training_settings import ValueTrainingSettings
from openmind.training.service.value_distiller import ValueDistiller

logger = logging.getLogger(__name__)


class ValueTrainingLoop:
    """Trains value rules round after round from self-play. In each round, self-play agents value the positions their
    rollouts reach with the previous round's rules, or the start rules, after a few rollout actions, and deduce the
    positions those rules have no clue about when the settings give a deduction budget; the round fits new rules; then the new rules' agent plays
    the random policy, untrained MCTS and the previous round's agent, every agent searching with its game's own seed.
    Round k seeds with the distillation's seed plus k. With a game memory, every game, self-play and evaluation alike, is
    remembered as it ends, with the model each player played."""

    def __init__(
        self,
        value_distiller: ValueDistiller,
        match_runner: MatchRunner,
        knowledge_base: KnowledgeBase,
        game_memory: GameMemory | None = None,
    ) -> None:
        self._game_memory = game_memory
        self._value_distiller = value_distiller
        self._match_runner = match_runner
        self._knowledge_base = knowledge_base

    def train(
        self,
        rbs: RuleBasedSystem,
        start: str | None,
        settings: ValueTrainingSettings,
        on_round: Callable[[TrainingReport], None] | None = None,
    ) -> TrainingReport:
        """Hands the report so far to on_round once a round's rules are fitted, before its games, so they can be saved
        even if the games never end, and again after every round; the last report is complete. To run in several
        workers, the agents playing a round's rules must pickle."""
        created_at = datetime.now().replace(microsecond=0)
        previous, previous_name = start, None if start is None else START_RULES
        rounds: list[TrainingRound] = []
        report = TrainingReport(rbs.context, created_at, settings, (), settings.rounds == 0)
        for number in range(1, settings.rounds + 1):
            started = time.perf_counter()
            if previous is None:
                logger.info("Round %d of %d: self-play without value rules", number, settings.rounds)
            else:
                logger.info(
                    "Round %d of %d: self-play valuing positions with %s",
                    number,
                    settings.rounds,
                    f"the {previous_name}" if previous_name == START_RULES else f"{previous_name}'s rules",
                )
            seed = settings.distillation.seed + number
            result = self._value_distiller.distill(
                rbs,
                self._agent_builder(rbs, previous, settings),
                replace(settings.distillation, seed=seed),
                self._round_declarer(rbs, number),
                number,
                "untrained" if previous_name is None else previous_name,
            )
            logger.info(
                "Round %d: %d value rules, held-out loss %s, held-out error %s",
                number,
                len(result.rules),
                None if result.chosen is None else result.chosen.held_out_loss,
                result.held_out_error,
            )
            baselines: tuple[MatchResults, ...] = ()
            against_previous: MatchResults | None = None
            if settings.evaluation_games and on_round is not None:
                fitted = TrainingRound(
                    number,
                    result.context,
                    result.rules,
                    result.fits,
                    result.chosen,
                    result.training_rows,
                    result.held_out_rows,
                    result.held_out_error,
                    (),
                    None,
                    time.perf_counter() - started,
                    result.records,
                )
                logger.info("Round %d fitted: handing it over before its games", number)
                on_round(TrainingReport(rbs.context, created_at, settings, (*rounds, fitted), False))
            if settings.evaluation_games:
                rng = random.Random(seed)
                evaluated = partial(create_seeded_agent, self._agent_builder(rbs, result.context, settings))
                evaluated_model = self._agent_builder(rbs, result.context, settings).describe(f"round {number}")
                untrained = self._agent_builder(rbs, None, settings, False)
                opponents: tuple[tuple[str, PolicyFactory, ModelDescription], ...] = (
                    (RANDOM_OPPONENT, create_random_policy, ModelDescription(RANDOM_OPPONENT, RANDOM_POLICY_TEXT)),
                    (UNTRAINED_OPPONENT, partial(create_seeded_agent, untrained), untrained.describe(UNTRAINED_OPPONENT)),
                )
                baselines = tuple(
                    self._series(
                        rbs,
                        number,
                        evaluated,
                        name,
                        opponent,
                        settings.evaluation_games,
                        rng,
                        evaluated_model,
                        model,
                        settings.distillation.time_control,
                    )
                    for name, opponent, model in opponents
                )
                if previous is not None and previous_name is not None:
                    previous_builder = self._agent_builder(rbs, previous, settings)
                    against_previous = self._series(
                        rbs,
                        number,
                        evaluated,
                        previous_name,
                        partial(create_seeded_agent, previous_builder),
                        settings.evaluation_games,
                        rng,
                        evaluated_model,
                        previous_builder.describe(previous_name),
                        settings.distillation.time_control,
                    )
            seconds = time.perf_counter() - started
            logger.info("Round %d took %.0f seconds", number, seconds)
            rounds.append(
                TrainingRound(
                    number,
                    result.context,
                    result.rules,
                    result.fits,
                    result.chosen,
                    result.training_rows,
                    result.held_out_rows,
                    result.held_out_error,
                    baselines,
                    against_previous,
                    seconds,
                    result.records,
                )
            )
            previous, previous_name = result.context, f"round {number}"
            report = TrainingReport(rbs.context, created_at, settings, tuple(rounds), number == settings.rounds)
            if on_round is not None:
                on_round(report)
        return report

    def _round_declarer(self, rbs: RuleBasedSystem, number: int) -> RuleDeclarer:
        """Where a round's fitted rules go: a variant of the game carrying its rules, so the round is a game of its
        own that can be played, evaluated and read back."""
        declarer = RuleDeclarer(self._knowledge_base, f"{rbs.context} round {number}")
        declarer.inherits(rbs.context)
        return declarer

    def _agent_builder(
        self, rbs: RuleBasedSystem, context: str | None, settings: ValueTrainingSettings, deduces: bool = True
    ) -> AgentBuilder:
        """An agent searching with the settings' iterations and rollout limit; with a value base, valuing the positions
        its rollouts reach after the settings' rollout actions; and, unless told not to, falling back on the settings'
        deduction when its rules have no clue. Untrained MCTS never deduces."""
        builder = AgentBuilder().with_exploration(EXPLORATION).with_iterations(settings.distillation.iterations)
        builder.with_rollout_limit(settings.rollout_limit, settings.unfinished_payoff)
        builder.with_deduction(settings.deduction if deduces else None)
        if settings.distillation.time_control is not None:
            builder.with_time_budget_estimator(
                PlainTimeBudgetEstimator(
                    settings.distillation.expected_steps,
                    settings.distillation.time_control.base_seconds * settings.distillation.time_reserve,
                )
            )
        distillation = settings.distillation
        valuer = None
        if context is not None:
            valuer = create_rule_based_system(self._knowledge_base, context)
            builder.with_valuation(valuer).with_rollout_actions(settings.rollout_actions)
        kind = VALUE_PRIOR if distillation.prior == VALUE_PRIOR and valuer is not None else UNIFORM_PRIOR
        builder.with_selection(distillation.selection, distillation.puct_exploration).with_prior(
            create_move_prior(kind, distillation.prior_temperature, rbs, None, valuer)
        )
        return builder

    def _series(
        self,
        rbs: RuleBasedSystem,
        number: int,
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
                        rbs,
                        game,
                        MATCH_GAME,
                        number,
                        index + 1,
                        seat,
                        evaluated_model,
                        opponent_model,
                        record,
                        time_control,
                    )
                )

            on_game = remember
        results = self._match_runner.series(rbs, evaluated, opponent, name, games, rng, time_control, on_game)
        logger.info(
            "Round %d against %s: %d games, %d wins, %d draws, %d losses",
            number,
            name,
            results.games,
            results.wins,
            results.draws,
            results.losses,
        )
        return results
