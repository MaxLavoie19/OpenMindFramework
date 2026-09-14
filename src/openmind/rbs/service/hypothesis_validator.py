from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.constant.generation_constant import PERMUTATION_BATCH
from openmind.rbs.mapper.action_row_mapper import ActionRowMapper
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.rbs.model.hypothesis import Hypothesis
from openmind.rbs.model.hypothesis_test import HypothesisTest
from openmind.rbs.service.advantage_contrast import AdvantageContrast
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rule.model.python_rule import PythonRule


class HypothesisValidator:
    """Tests discovered hypotheses on validation rows, which must come from other games than the discovery rows.

    The unit is the state: for each validation state where, within the hypothesis's parent scope, both matching and
    other actions occur, the difference of their visit-weighted mean advantages (see AdvantageContrast). Under the null
    hypothesis the condition doesn't tell the two apart, so each difference is as likely to have either sign; a
    one-sided sign-flip permutation test in the discovered direction gives the p-value, (1 + permutations at least as
    extreme) / (1 + permutations). The p-values of every hypothesis tested together are adjusted with Benjamini-Hochberg,
    and a hypothesis is validated when its q-value is within the false discovery rate and its validation effect points
    the discovered way. A hypothesis without any validation state gets a p-value of 1."""

    def __init__(
        self,
        condition_evaluator: ConditionEvaluator,
        action_row_mapper: ActionRowMapper,
        advantage_contrast: AdvantageContrast,
    ) -> None:
        self._condition_evaluator = condition_evaluator
        self._action_row_mapper = action_row_mapper
        self._advantage_contrast = advantage_contrast

    def validate(
        self,
        domain: Domain,
        hypotheses: Sequence[Hypothesis],
        rows: Sequence[ActionRow],
        settings: GenerationSettings,
        seed: int,
    ) -> tuple[HypothesisTest, ...]:
        rng = np.random.default_rng(seed)
        by_action: dict[str, list[ActionRow]] = {}
        for row in rows:
            by_action.setdefault(row.action.name, []).append(row)
        masks: dict[tuple[str, PythonRule], np.ndarray | None] = {}

        def mask(action: str, condition: PythonRule) -> np.ndarray | None:
            key = (action, condition)
            if key not in masks:
                masks[key] = self._condition_evaluator.mask(domain, by_action[action], condition)
            return masks[key]

        measured: list[tuple[float | None, int, float]] = []
        for hypothesis in hypotheses:
            action_rows = by_action.get(hypothesis.action, [])
            condition_masks = [mask(hypothesis.action, condition) for condition in hypothesis.conditions] if action_rows else []
            parent_masks = [mask(hypothesis.action, condition) for condition in hypothesis.parent] if action_rows else []
            if not action_rows or any(value is None for value in (*condition_masks, *parent_masks)):
                measured.append((None, 0, 1.0))
                continue
            arrays = self._action_row_mapper.to_arrays(action_rows)
            everything = np.ones(len(action_rows), dtype=bool)
            scope = np.logical_and.reduce([everything, *parent_masks])  # type: ignore[list-item]
            matching = np.logical_and.reduce([everything, *condition_masks])  # type: ignore[list-item]
            differences = self._advantage_contrast.differences(arrays, scope, matching)
            if len(differences) == 0:
                measured.append((None, 0, 1.0))
                continue
            effect = float(differences.mean())
            measured.append((effect, len(differences), self._p_value(differences, hypothesis.direction, settings.permutations, rng)))

        q_values = self._q_values([p_value for _, _, p_value in measured])
        return tuple(
            HypothesisTest(
                hypothesis.action,
                hypothesis.conditions,
                hypothesis.direction,
                hypothesis.effect,
                hypothesis.states,
                effect,
                states,
                p_value,
                q_value,
                effect is not None and effect * hypothesis.direction > 0 and q_value <= settings.false_discovery_rate,
            )
            for hypothesis, (effect, states, p_value), q_value in zip(hypotheses, measured, q_values, strict=True)
        )

    def _p_value(self, differences: np.ndarray, direction: int, permutations: int, rng: np.random.Generator) -> float:
        if len(differences) < 2:
            return 1.0
        observed = direction * float(differences.mean())
        extreme = 0
        remaining = permutations
        while remaining > 0:
            batch = min(PERMUTATION_BATCH, remaining)
            signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(differences)))
            flipped = direction * (signs * differences).mean(axis=1)
            extreme += int(np.count_nonzero(flipped >= observed - 1e-12))
            remaining -= batch
        return (1 + extreme) / (1 + permutations)

    def _q_values(self, p_values: list[float]) -> list[float]:
        """Benjamini-Hochberg adjusted p-values, in the given order."""
        count = len(p_values)
        if count == 0:
            return []
        order = np.argsort(p_values, kind="stable")
        ranked = np.array(p_values)[order] * count / np.arange(1, count + 1)
        adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
        q_values = np.empty(count)
        q_values[order] = np.minimum(adjusted, 1.0)
        return [float(value) for value in q_values]
