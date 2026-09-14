import logging
import math
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.mapper.action_row_mapper import ActionRowMapper
from openmind.rbs.mapper.hypothesis_text_mapper import HypothesisTextMapper
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.coverage import Coverage
from openmind.rbs.model.generation_result import GenerationResult
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.rbs.model.hypothesis import Hypothesis
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.coverage_filter import CoverageFilter
from openmind.rbs.service.goal_pattern_miner import GoalPatternMiner
from openmind.rbs.service.hypothesis_discoverer import HypothesisDiscoverer
from openmind.rbs.service.hypothesis_validator import HypothesisValidator
from openmind.rbs.service.primitive_generator import PrimitiveGenerator

logger = logging.getLogger(__name__)


class RuleGenerator:
    """An inference engine for rules, for any domain. From the search samples of discovery games it mines goal patterns,
    generates primitives and discovers hypotheses about which actions deserve more or less exploration; it tests them on
    the samples of validation games and keeps those that hold. The rule base has, for each action, a rule without
    conditions, its mean payoff, then one rule per validated hypothesis that no simpler rule covers (see
    CoverageFilter)."""

    def __init__(
        self,
        action_row_mapper: ActionRowMapper,
        goal_pattern_miner: GoalPatternMiner,
        primitive_generator: PrimitiveGenerator,
        hypothesis_discoverer: HypothesisDiscoverer,
        hypothesis_validator: HypothesisValidator,
        coverage_filter: CoverageFilter,
        rule_text_mapper: RuleTextMapper,
        hypothesis_text_mapper: HypothesisTextMapper,
    ) -> None:
        self._action_row_mapper = action_row_mapper
        self._goal_pattern_miner = goal_pattern_miner
        self._primitive_generator = primitive_generator
        self._hypothesis_discoverer = hypothesis_discoverer
        self._hypothesis_validator = hypothesis_validator
        self._coverage_filter = coverage_filter
        self._rule_text_mapper = rule_text_mapper
        self._hypothesis_text_mapper = hypothesis_text_mapper

    def generate(
        self,
        domain: Domain,
        discovery: Sequence[ActionSample],
        validation: Sequence[ActionSample],
        settings: GenerationSettings,
        seed: int,
    ) -> GenerationResult:
        discovery_rows = self._action_row_mapper.to_rows(discovery, settings.min_visits)
        validation_rows = self._action_row_mapper.to_rows(validation, settings.min_visits)
        patterns = self._goal_pattern_miner.patterns(domain, discovery_rows, settings.patterns)

        bases: list[Rule] = []
        rows_by_action: dict[str, list[ActionRow]] = {}
        hypotheses: list[Hypothesis] = []
        for name in dict.fromkeys(row.action.name for row in discovery_rows):
            rows = [row for row in discovery_rows if row.action.name == name]
            rows_by_action[name] = rows
            visits = sum(row.visits for row in rows)
            bases.append(Rule(name, (), math.fsum(row.visits * row.mean_payoff for row in rows) / visits, visits))
            primitives = self._primitive_generator.primitives(domain, rows, patterns, settings)
            found = self._hypothesis_discoverer.discover(domain, name, rows, primitives, patterns, settings)
            logger.info(
                "%s: %d primitives, %d hypotheses discovered from %d rows in %d states",
                name,
                len(primitives),
                len(found),
                len(rows),
                len({row.state for row in rows}),
            )
            hypotheses.extend(found)

        tests = self._hypothesis_validator.validate(domain, hypotheses, validation_rows, settings, seed)
        validated = [hypothesis for hypothesis, test in zip(hypotheses, tests, strict=True) if test.validated]
        logger.info(
            "Validated %d of %d hypotheses on %d rows in %d states at a false discovery rate of %s",
            len(validated),
            len(tests),
            len(validation_rows),
            len({row.state for row in validation_rows}),
            settings.false_discovery_rate,
        )
        if logger.isEnabledFor(logging.DEBUG):
            for test in tests:
                logger.debug("%s", self._hypothesis_text_mapper.to_text(test))

        rules: list[Rule] = []
        covered: list[Coverage] = []
        for base in bases:
            candidates = (
                base,
                *(
                    Rule(hypothesis.action, hypothesis.conditions, hypothesis.expected_value, hypothesis.visits, hypothesis.priority)
                    for hypothesis in validated
                    if hypothesis.action == base.action
                ),
            )
            if settings.coverage:
                kept, left_out = self._coverage_filter.keep(domain, rows_by_action[base.action], candidates, settings.min_gain)
            else:
                kept, left_out = candidates, ()
            logger.info(
                "%s: kept %d of %d validated rules; %d covered by a simpler rule within min_gain",
                base.action,
                len(kept) - 1,
                len(candidates) - 1,
                len(left_out),
            )
            if logger.isEnabledFor(logging.DEBUG):
                for coverage in left_out:
                    logger.debug(
                        "Covered: %s by %s",
                        self._rule_text_mapper.to_text(coverage.rule),
                        self._rule_text_mapper.to_text(coverage.covering),
                    )
            rules.extend(kept)
            covered.extend(left_out)
        if logger.isEnabledFor(logging.DEBUG):
            for rule in rules:
                logger.debug("%s", self._rule_text_mapper.to_text(rule))
        return GenerationResult(RuleBase(domain.name, tuple(rules)), patterns, tests, tuple(covered))
