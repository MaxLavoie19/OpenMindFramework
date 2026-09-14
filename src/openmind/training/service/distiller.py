import logging
import math
import random

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.rule_generator import RuleGenerator
from openmind.rbs.service.rule_rater import RuleRater
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.model.distillation_result import DistillationResult
from openmind.training.model.distillation_settings import DistillationSettings
from openmind.training.service.self_play import SelfPlay

logger = logging.getLogger(__name__)


class Distiller:
    """Distills a rule base from self-play: generates hypotheses from training games, validates them on held-out games,
    and measures the kept rules on those held-out games."""

    def __init__(
        self,
        self_play: SelfPlay,
        rule_generator: RuleGenerator,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
    ) -> None:
        self._self_play = self_play
        self._rule_generator = rule_generator
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library

    def distill(
        self, domain: Domain, agent_builder: AgentBuilder, settings: DistillationSettings
    ) -> DistillationResult:
        """Sets the builder's iterations; the builder needs its exploration set."""
        rng = random.Random(settings.seed)
        agent_builder.with_iterations(settings.iterations)
        training = self._self_play.play(domain, agent_builder, settings.games, rng)
        held_out = self._self_play.play(domain, agent_builder, settings.held_out_games, rng)
        generation = self._rule_generator.generate(domain, training, held_out, settings.generation, settings.seed)
        rule_base = generation.rule_base
        rating_error = self._rating_error(domain, rule_base, held_out, settings.generation.min_visits)
        rules = rule_base.rules
        mean_conditions = math.fsum(len(rule.conditions) for rule in rules) / len(rules) if rules else 0.0
        logger.info(
            "Distilled %d rules, %s conditions per rule on average, from %d training samples; %d of %d hypotheses "
            "validated, %d covered by a simpler rule, and rating error %s on %d held-out samples",
            len(rules),
            mean_conditions,
            len(training),
            sum(test.validated for test in generation.hypotheses),
            len(generation.hypotheses),
            len(generation.covered),
            rating_error,
            len(held_out),
        )
        return DistillationResult(
            rule_base,
            len(training),
            len(held_out),
            rating_error,
            mean_conditions,
            generation.patterns,
            generation.hypotheses,
            generation.covered,
        )

    def _rating_error(
        self, domain: Domain, rule_base: RuleBase, samples: tuple[ActionSample, ...], min_visits: int
    ) -> float | None:
        rater = RuleRater(rule_base, domain, self._rule_compiler, self._rule_runner, self._consequence_library)
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
