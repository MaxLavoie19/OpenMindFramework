from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.coverage import Coverage
from openmind.rbs.model.rule import Rule
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rule.model.python_rule import PythonRule


class CoverageFilter:
    """Leaves out the rules a simpler rule already covers, judged on the rows the rules were discovered from. A rule is
    covered by a kept rule that has no more conditions, is a priority rule whenever the rule is one, matches every row
    the rule matches, and whose rows' visit-weighted mean advantage is within min_gain of the rule's rows'. Advantage, a
    row's mean payoff minus the best in its state, compares actions within their own positions, so how good those
    positions are cancels out, where expected values would mix the two. Rules are checked fewest conditions first, then
    most rows matched, so a rule can only be covered by one checked, and kept, before it. A rule whose conditions can't
    be evaluated on every row, or that matches no row, is kept and covers nothing."""

    def __init__(self, condition_evaluator: ConditionEvaluator) -> None:
        self._condition_evaluator = condition_evaluator

    def keep(
        self, domain: Domain, rows: Sequence[ActionRow], rules: Sequence[Rule], min_gain: float
    ) -> tuple[tuple[Rule, ...], tuple[Coverage, ...]]:
        """One action's rows and rules; the kept rules and the coverages, both in the given order."""
        unique = list(dict.fromkeys(condition for rule in rules for condition in rule.conditions))
        conditions = dict(zip(unique, self._condition_evaluator.masks(domain, rows, unique), strict=True))
        visits = np.array([row.visits for row in rows], dtype=float)
        advantages = np.array([row.advantage for row in rows], dtype=float)
        masks = [self._mask(len(rows), rule, conditions) for rule in rules]
        means = [self._mean_advantage(mask, visits, advantages) for mask in masks]
        order = sorted(
            range(len(rules)),
            key=lambda index: (
                len(rules[index].conditions),
                -(0 if (mask := masks[index]) is None else int(mask.sum())),
                index,
            ),
        )
        kept: list[int] = []
        covering: dict[int, int] = {}
        for index in order:
            found = (
                None
                if means[index] is None
                else next((other for other in kept if self._covers(other, index, rules, masks, means, min_gain)), None)
            )
            if found is None:
                kept.append(index)
            else:
                covering[index] = found
        return (
            tuple(rule for index, rule in enumerate(rules) if index not in covering),
            tuple(Coverage(rules[index], rules[found]) for index, found in sorted(covering.items())),
        )

    def _mask(self, rows: int, rule: Rule, conditions: dict[PythonRule, np.ndarray | None]) -> np.ndarray | None:
        mask = np.ones(rows, dtype=bool)
        for condition in rule.conditions:
            matching = conditions[condition]
            if matching is None:
                return None
            mask &= matching
        return mask

    def _mean_advantage(self, mask: np.ndarray | None, visits: np.ndarray, advantages: np.ndarray) -> float | None:
        if mask is None:
            return None
        weights = visits[mask]
        total = float(weights.sum())
        return float((weights * advantages[mask]).sum()) / total if total > 0 else None

    def _covers(
        self,
        simpler: int,
        index: int,
        rules: Sequence[Rule],
        masks: list[np.ndarray | None],
        means: list[float | None],
        min_gain: float,
    ) -> bool:
        simpler_mask, mask = masks[simpler], masks[index]
        simpler_mean, mean = means[simpler], means[index]
        return (
            simpler_mask is not None
            and mask is not None
            and simpler_mean is not None
            and mean is not None
            and len(rules[simpler].conditions) <= len(rules[index].conditions)
            and (rules[simpler].priority or not rules[index].priority)
            and not np.any(mask & ~simpler_mask)
            and abs(mean - simpler_mean) < min_gain
        )
