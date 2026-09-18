import logging
import math
import random
from collections.abc import Callable, Sequence

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import HELD_OUT_SELF_PLAY_GAME, SELF_PLAY_GAME
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.training.mapper.played_game_summary_mapper import PlayedGameSummaryMapper
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.value_distillation_result import ValueDistillationResult
from openmind.training.model.value_distillation_settings import ValueDistillationSettings
from openmind.training.service.self_play import SelfPlay
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class ValueDistiller:
    """Distills value rules from self-play: turns the positions of training and held-out games into rows valued at the
    target, fits value rules on the training rows, chooses among the fits on the held-out rows, and measures the chosen
    rules on the held-out rows. With a game memory, every game is remembered as it ends, with the model each player
    played."""

    def __init__(
        self,
        self_play: SelfPlay,
        value_generator: ValueGenerator,
        position_row_mapper: PositionRowMapper,
        knowledge_base: KnowledgeBase,
        game_memory: GameMemory | None = None,
        played_game_summary_mapper: PlayedGameSummaryMapper | None = None,
    ) -> None:
        self._game_memory = game_memory
        self._summaries = PlayedGameSummaryMapper() if played_game_summary_mapper is None else played_game_summary_mapper
        self._self_play = self_play
        self._value_generator = value_generator
        self._position_row_mapper = position_row_mapper
        self._knowledge_base = knowledge_base

    def distill(
        self,
        rbs: RuleBasedSystem,
        agent_builder: AgentBuilder,
        settings: ValueDistillationSettings,
        declarer: RuleDeclarer,
        round_number: int | None = None,
        model_name: str = SELF_PLAY_GAME,
    ) -> ValueDistillationResult:
        """Sets the builder's iterations; the builder needs its exploration set. `round_number` and `model_name`, what
        the self-play agent is called, name the games a game memory remembers."""
        rng = random.Random(settings.seed)
        agent_builder.with_iterations(settings.iterations)
        if settings.time_control is not None:
            agent_builder.with_time_budget_estimator(PlainTimeBudgetEstimator(settings.expected_steps, self._reserve(settings)))
        model = agent_builder.describe(model_name)
        players = len(rbs.players().names)
        training_games = self._self_play.play(
            rbs,
            agent_builder,
            settings.games,
            rng,
            keep_samples=False,
            time_control=settings.time_control,
            on_game=self._remembering(rbs, SELF_PLAY_GAME, round_number, lambda game: (model,) * players),
        )
        held_out_games = self._self_play.play(
            rbs,
            agent_builder,
            settings.held_out_games,
            rng,
            keep_samples=False,
            time_control=settings.time_control,
            on_game=self._remembering(rbs, HELD_OUT_SELF_PLAY_GAME, round_number, lambda game: (model,) * players),
        )
        training = self._position_row_mapper.to_rows(rbs, training_games, settings.target)
        held_out = self._position_row_mapper.to_rows(rbs, held_out_games, settings.target)
        generation = self._value_generator.generate(rbs, training, held_out, settings.values, declarer)
        held_out_error = self._error(declarer.context, held_out)
        logger.info(
            "Distilled %d value rules from %d training rows valued at the %s target; mean absolute error %s on %d "
            "held-out rows",
            len(generation.rules),
            len(training),
            settings.target,
            held_out_error,
            len(held_out),
        )
        return ValueDistillationResult(
            generation.context,
            generation.rules,
            generation.fits,
            generation.chosen,
            generation.candidates,
            len(training),
            len(held_out),
            held_out_error,
            records=self._self_play.records(rbs, training_games),
        )

    def _reserve(self, settings: ValueDistillationSettings) -> float:
        """The seconds of the base time an agent keeps in reserve on the settings' clock; 0 without one."""
        return 0.0 if settings.time_control is None else settings.time_control.base_seconds * settings.time_reserve

    def _remembering(
        self,
        rbs: RuleBasedSystem,
        kind: str,
        round_number: int | None,
        models: Callable[[PlayedGame], tuple[ModelDescription, ...]],
    ) -> Callable[[int, PlayedGame], None] | None:
        """What remembers each game as it ends, the models each player played given by `models`; None without a game
        memory."""
        memory = self._game_memory
        if memory is None:
            return None

        def remember(index: int, game: PlayedGame) -> None:
            record = next(iter(self._self_play.records(rbs, (game,))), None)
            memory.remember(self._summaries.to_summary(rbs, game, kind, round_number, index + 1, models(game), record))

        return remember

    def _error(self, context: str, rows: Sequence[PositionRow]) -> float | None:
        """The mean absolute difference between what the declared rules value a position at and the row's target; None
        where no held-out row could be valued."""
        valued = create_rule_based_system(self._knowledge_base, context)
        values_by_state: dict[State, tuple[float, ...] | None] = {}
        errors: list[float] = []
        for row in rows:
            if row.state not in values_by_state:
                values_by_state[row.state] = valued.values(row.state)
            values = values_by_state[row.state]
            if values is not None:
                errors.append(abs(values[valued.players().names.index(row.player)] - row.target))
        return math.fsum(errors) / len(errors) if errors else None
