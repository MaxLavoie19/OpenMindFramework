import logging
import math
import random
import time
from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.model.domain import Domain
from openmind.evaluation.constant.evaluation_constant import REFERENCE_TOLERANCE
from openmind.evaluation.model.action_values import ActionValues
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.choice_measure import ChoiceMeasure
from openmind.evaluation.service.choice_measurer import ChoiceMeasurer
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.evaluation.service.reference_search import ReferenceSearch
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.factory.rbs_factory import create_rule_rater
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.training.constant.training_constant import CONFIRMATION_SEED_OFFSET, SELECTION_KIND
from openmind.training.model.non_inferiority import NonInferiority
from openmind.training.model.removal_test import RemovalTest
from openmind.training.model.selection_report import SelectionReport
from openmind.training.model.selection_settings import SelectionSettings
from openmind.training.service.non_inferiority_test import NonInferiorityTest
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class RuleSelector:
    """Selects the smallest set of candidate rules that plays no worse than all of them.

    An agent guided by every candidate searches each selection position. Then, pass after pass, each rule with
    conditions is left out in turn, least used first (the fewest ratings decided on the positions, then the fewest
    visits), and stays out when the positions searched without it show a rise in mean regret whose upper bound is below
    the margin (see NonInferiorityTest). Every comparison is against the full set, so small losses don't add up. When the
    positions are every position exact search finds, rules that decide no rating are removed without a search: no search
    reads a rating they give. Rules without conditions are always kept. Passes stop when one removes nothing, or when
    the time is up. Finally, the selected set is confirmed against the full set with the next seed, on new positions
    where they are sampled. Searches run in the task runner's workers, the positions split between them."""

    def __init__(
        self,
        exact_search: ExactSearch,
        reference_search: ReferenceSearch,
        choice_measurer: ChoiceMeasurer,
        task_runner: TaskRunner,
        non_inferiority_test: NonInferiorityTest,
        rule_text_mapper: RuleTextMapper,
    ) -> None:
        self._exact_search = exact_search
        self._reference_search = reference_search
        self._choice_measurer = choice_measurer
        self._task_runner = task_runner
        self._non_inferiority_test = non_inferiority_test
        self._rule_text_mapper = rule_text_mapper

    def select(
        self,
        domain: Domain,
        candidates: RuleBase,
        candidates_file: str,
        settings: SelectionSettings,
        on_progress: Callable[[SelectionReport], None] | None = None,
    ) -> SelectionReport:
        """Calls on_progress with the report so far after every decision. A reference search samples positions, so it
        needs a number of positions rather than all of them."""
        created_at = datetime.now().replace(microsecond=0)
        deadline = None if settings.max_hours is None else time.monotonic() + settings.max_hours * 3600
        rng = np.random.default_rng(settings.seed)
        sample, values, tolerance = self._positions(domain, settings, random.Random(settings.seed), settings.seed)
        every_position = settings.reference_iterations is None and settings.positions is None
        full_measures = self._measure(domain, candidates.rules, sample, values, tolerance, settings, settings.seed)
        full_regrets = self._regrets(full_measures)
        full_optimal = sum(measure.optimal for measure in full_measures)
        full = self._agreement(full_measures, settings.iterations)
        logger.info(
            "All %d rules at %d iterations on %d positions: %d optimal choices, mean regret %s",
            len(candidates.rules),
            settings.iterations,
            full.positions,
            full.optimal,
            full.mean_regret,
        )
        kept = list(candidates.rules)
        subset = full
        tests: list[RemovalTest] = []
        complete = True

        def report(confirmation: NonInferiority | None = None) -> SelectionReport:
            return SelectionReport(
                domain.name,
                created_at,
                candidates_file,
                settings,
                len(candidates.rules),
                RuleBase(candidates.domain, tuple(kept)),
                full,
                subset,
                confirmation,
                tuple(tests),
                complete,
            )

        pass_number = 0
        while True:
            pass_number += 1
            removed = 0
            usage = self._usage(domain, kept, sample, values)
            if every_position:
                unused = [rule for rule in kept if rule.conditions and usage[rule] == 0]
                for rule in unused:
                    kept.remove(rule)
                    tests.append(RemovalTest(rule, pass_number, 0.0, 0.0, 0, True, True, 0.0))
                    logger.info("Removed %s: it decides no rating", self._rule_text_mapper.to_text(rule))
                removed += len(unused)
                if unused and on_progress is not None:
                    on_progress(report())
            for rule in sorted((rule for rule in kept if rule.conditions), key=lambda rule: (usage[rule], rule.visits)):
                if deadline is not None and time.monotonic() > deadline:
                    complete = False
                    logger.info("Out of time after %s hours", settings.max_hours)
                    break
                started = time.perf_counter()
                trial = [candidate for candidate in kept if candidate != rule]
                measures = self._measure(domain, trial, sample, values, tolerance, settings, settings.seed)
                comparison = self._non_inferiority_test.test(
                    self._regrets(measures) - full_regrets, settings.margin, settings.confidence, settings.resamples, rng
                )
                optimal_difference = sum(measure.optimal for measure in measures) - full_optimal
                seconds = time.perf_counter() - started
                tests.append(
                    RemovalTest(
                        rule,
                        pass_number,
                        comparison.regret_difference,
                        comparison.upper_bound,
                        optimal_difference,
                        comparison.holds,
                        False,
                        seconds,
                    )
                )
                if comparison.holds:
                    kept = trial
                    subset = self._agreement(measures, settings.iterations)
                    removed += 1
                logger.info(
                    "%s %s: regret %+f (bound %s %s margin %s), optimal choices %+d, %.1f seconds",
                    "Removed" if comparison.holds else "Kept",
                    self._rule_text_mapper.to_text(rule),
                    comparison.regret_difference,
                    comparison.upper_bound,
                    "<" if comparison.holds else ">=",
                    settings.margin,
                    optimal_difference,
                    seconds,
                )
                if on_progress is not None:
                    on_progress(report())
            logger.info(
                "Pass %d removed %d rules: %d of %d kept", pass_number, removed, len(kept), len(candidates.rules)
            )
            if not complete or removed == 0:
                break

        confirmation = self._confirm(domain, candidates.rules, kept, settings, every_position, (sample, values, tolerance))
        final = report(confirmation)
        if on_progress is not None:
            on_progress(final)
        return final

    def _confirm(
        self,
        domain: Domain,
        candidates: Sequence[Rule],
        kept: Sequence[Rule],
        settings: SelectionSettings,
        every_position: bool,
        positions: tuple[list[State], list[ActionValues], float],
    ) -> NonInferiority:
        seed = settings.seed + CONFIRMATION_SEED_OFFSET
        sample, values, tolerance = positions if every_position else self._positions(domain, settings, random.Random(seed), seed)
        full = self._measure(domain, candidates, sample, values, tolerance, settings, seed)
        subset = self._measure(domain, kept, sample, values, tolerance, settings, seed)
        confirmation = self._non_inferiority_test.test(
            self._regrets(subset) - self._regrets(full),
            settings.margin,
            settings.confidence,
            settings.resamples,
            np.random.default_rng(seed),
        )
        logger.info(
            "Confirmation with seed %d on %d positions: regret %+f, bound %s %s margin %s",
            seed,
            confirmation.positions,
            confirmation.regret_difference,
            confirmation.upper_bound,
            "<" if confirmation.holds else ">=",
            settings.margin,
        )
        return confirmation

    def _positions(
        self, domain: Domain, settings: SelectionSettings, rng: random.Random, seed: int
    ) -> tuple[list[State], list[ActionValues], float]:
        """The positions, every legal action's value in each, and how far from the best an optimal action's value may
        be."""
        if settings.reference_iterations is None:
            positions = self._exact_search.positions(domain)
            if settings.positions is None:
                sample = list(positions)
            else:
                sample = rng.sample(positions, min(settings.positions, len(positions)))
            return sample, [self._exact_search.action_values(domain, state) for state in sample], 0.0
        if settings.positions is None:
            raise ValueError("A reference search samples positions: give a number of positions, not all of them")
        sample = list(self._reference_search.positions(domain, settings.positions, rng))
        count = len(sample)
        values = self._task_runner.map(
            self._reference_search.action_values,
            [domain] * count,
            sample,
            [settings.reference_iterations] * count,
            [seed] * count,
        )
        return sample, values, REFERENCE_TOLERANCE

    def _measure(
        self,
        domain: Domain,
        rules: Sequence[Rule],
        sample: list[State],
        values: list[ActionValues],
        tolerance: float,
        settings: SelectionSettings,
        seed: int,
    ) -> list[ChoiceMeasure]:
        """Every position searched by an agent the rules guide, in order."""
        rater = create_rule_rater(RuleBase(domain.name, tuple(rules)), domain)
        builder = (
            AgentBuilder()
            .with_exploration(EXPLORATION)
            .with_guidance(rater)
            .with_guided_rollouts(settings.guided_rollouts)
            .with_iterations(settings.iterations)
            .with_seed(seed)
        )
        slices = self._task_runner.split(list(zip(sample, values, strict=True)))
        count = len(slices)
        results = self._task_runner.map(
            self._choice_measurer.measure,
            [domain] * count,
            [builder] * count,
            slices,
            [tolerance] * count,
            [SELECTION_KIND] * count,
            [settings.iterations] * count,
        )
        return [measure for result in results for measure in result]

    def _usage(
        self, domain: Domain, rules: Sequence[Rule], sample: list[State], values: list[ActionValues]
    ) -> dict[Rule, int]:
        """How many ratings, over every legal action of every position, each rule decides."""
        rater = create_rule_rater(RuleBase(domain.name, tuple(rules)), domain)
        usage = dict.fromkeys(rules, 0)
        for state, action_values in zip(sample, values, strict=True):
            for action, _ in action_values:
                if (rule := rater.explain(state, action)) is not None:
                    usage[rule] += 1
        return usage

    def _regrets(self, measures: Sequence[ChoiceMeasure]) -> np.ndarray:
        return np.array([measure.regret for measure in measures], dtype=float)

    def _agreement(self, measures: Sequence[ChoiceMeasure], iterations: int) -> Agreement:
        count = len(measures)
        return Agreement(
            iterations,
            count,
            sum(measure.optimal for measure in measures),
            math.fsum(measure.optimal_visit_share for measure in measures) / count if count else 0.0,
            math.fsum(measure.regret for measure in measures) / count if count else 0.0,
            math.fsum(measure.seconds for measure in measures) / count if count else 0.0,
        )
