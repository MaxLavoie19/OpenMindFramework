import logging
import math
import random

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.expression.service.interpreter import Interpreter
from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_inducer import RuleInducer
from openmind.rbs.service.rule_rater import RuleRater
from openmind.training.model.distillation_result import DistillationResult
from openmind.training.model.distillation_settings import DistillationSettings
from openmind.training.service.self_play import SelfPlay

logger = logging.getLogger(__name__)


class Distiller:
    """Distills a rule base from self-play: induces rules from training games and measures them on held-out games."""

    def __init__(self, self_play: SelfPlay, rule_inducer: RuleInducer, interpreter: Interpreter) -> None:
        self._self_play = self_play
        self._rule_inducer = rule_inducer
        self._interpreter = interpreter

    def distill(
        self, domain: Domain, agent_builder: AgentBuilder, settings: DistillationSettings
    ) -> DistillationResult:
        """Sets the builder's iterations; the builder needs its exploration set."""
        rng = random.Random(settings.seed)
        agent_builder.with_iterations(settings.iterations)
        training = self._self_play.play(domain, agent_builder, settings.games, rng)
        held_out = self._self_play.play(domain, agent_builder, settings.held_out_games, rng)
        rule_base = self._rule_inducer.induce(domain.name, training, settings.induction)
        rating_error = self._rating_error(rule_base, held_out, settings.induction.min_visits)
        rules = rule_base.rules
        mean_conditions = math.fsum(len(rule.conditions) for rule in rules) / len(rules) if rules else 0.0
        logger.info(
            "Distilled %d rules, %s conditions per rule on average, from %d training samples; "
            "rating error %s on %d held-out samples",
            len(rules),
            mean_conditions,
            len(training),
            rating_error,
            len(held_out),
        )
        return DistillationResult(rule_base, len(training), len(held_out), rating_error, mean_conditions)

    def _rating_error(self, rule_base: RuleBase, samples: tuple[ActionSample, ...], min_visits: int) -> float | None:
        rater = RuleRater(rule_base, self._interpreter)
        visits = 0
        error = 0.0
        for sample in samples:
            if sample.visits < min_visits:
                continue
            (rating,) = rater.rate(sample.state, (sample.action,))
            if rating is None:
                continue
            visits += sample.visits
            error += sample.visits * abs(rating - sample.mean_payoff)
        return error / visits if visits else None
