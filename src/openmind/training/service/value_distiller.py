import logging
import math
import random
from collections.abc import Sequence

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.value_distillation_result import ValueDistillationResult
from openmind.training.model.value_distillation_settings import ValueDistillationSettings
from openmind.training.service.self_play import SelfPlay
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class ValueDistiller:
    """Distills value rules from self-play: turns the positions of training and held-out games into rows valued at the
    target, fits value rules on the training rows, chooses among the fits on the held-out rows, and measures the chosen
    rules on the held-out rows."""

    def __init__(
        self,
        self_play: SelfPlay,
        value_generator: ValueGenerator,
        position_row_mapper: PositionRowMapper,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
    ) -> None:
        self._self_play = self_play
        self._value_generator = value_generator
        self._position_row_mapper = position_row_mapper
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library

    def distill(
        self, domain: Domain, agent_builder: AgentBuilder, settings: ValueDistillationSettings
    ) -> ValueDistillationResult:
        """Sets the builder's iterations; the builder needs its exploration set."""
        rng = random.Random(settings.seed)
        agent_builder.with_iterations(settings.iterations)
        training_games = self._self_play.play(domain, agent_builder, settings.games, rng)
        held_out_games = self._self_play.play(domain, agent_builder, settings.held_out_games, rng)
        training = self._position_row_mapper.to_rows(domain, training_games, settings.target)
        held_out = self._position_row_mapper.to_rows(domain, held_out_games, settings.target)
        generation = self._value_generator.generate(domain, training, held_out, settings.values)
        held_out_error = self._error(domain, generation.value_base, held_out)
        logger.info(
            "Distilled %d value rules from %d training rows valued at the %s target; mean absolute error %s on %d "
            "held-out rows",
            len(generation.value_base.rules),
            len(training),
            settings.target,
            held_out_error,
            len(held_out),
        )
        return ValueDistillationResult(
            generation.value_base,
            generation.fits,
            generation.chosen,
            generation.candidates,
            len(training),
            len(held_out),
            held_out_error,
        )

    def _error(self, domain: Domain, value_base: ValueBase, rows: Sequence[PositionRow]) -> float | None:
        valuer = RuleValuer(value_base, domain, self._rule_compiler, self._rule_runner, self._consequence_library)
        values_by_state: dict[State, tuple[float, ...] | None] = {}
        errors: list[float] = []
        for row in rows:
            if row.state not in values_by_state:
                values_by_state[row.state] = valuer.value(row.state)
            values = values_by_state[row.state]
            if values is not None:
                errors.append(abs(values[domain.players.names.index(row.player)] - row.target))
        return math.fsum(errors) / len(errors) if errors else None
