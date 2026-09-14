import logging
import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.constant.generation_constant import MIN_STATES, PRIORITY_MARGIN
from openmind.rbs.mapper.action_row_mapper import ActionRowMapper
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.model.hypothesis import Hypothesis
from openmind.rbs.model.row_arrays import RowArrays
from openmind.rbs.service.advantage_contrast import AdvantageContrast
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rule.model.python_rule import PythonRule

logger = logging.getLogger(__name__)


class HypothesisDiscoverer:
    """Discovers hypotheses about which of an action's forms deserve more or less exploration, from discovery rows.

    Every primitive is evaluated on the rows. One that is true for some actions of a state and false for others can
    split exploration: within a scope, it is compared state by state with the other actions (see AdvantageContrast), and
    becomes a hypothesis when its matching actions have min_rule_visits visits, the comparison covers MIN_STATES states,
    and the mean difference is at least min_gain. One that is the same for every action of a state can't split
    exploration by itself, but can gate: it is tried as the scope of the others. The beam_width hypotheses of each size
    with the highest score, the mean difference over its standard error, are kept and extended with one more
    primitive, up to max_conditions conditions. Each goal pattern is also tried whole, as one hypothesis.

    No hypothesis has a condition that changes nothing on the rows: a combination is skipped when one of its conditions
    can be dropped without changing the rows it matches, and a goal pattern's conditions are dropped, first to last,
    while the rows it matches stay the same. Hypotheses matching the same rows are merged into the one with the fewest
    conditions, then the highest score."""

    def __init__(
        self,
        condition_evaluator: ConditionEvaluator,
        action_row_mapper: ActionRowMapper,
        advantage_contrast: AdvantageContrast,
    ) -> None:
        self._condition_evaluator = condition_evaluator
        self._action_row_mapper = action_row_mapper
        self._advantage_contrast = advantage_contrast

    def discover(
        self,
        domain: Domain,
        action: str,
        rows: Sequence[ActionRow],
        primitives: Sequence[PythonRule],
        patterns: Sequence[GoalPattern],
        settings: GenerationSettings,
    ) -> tuple[Hypothesis, ...]:
        if not rows:
            return ()
        arrays = self._action_row_mapper.to_arrays(rows)
        counts = np.bincount(arrays.states)
        splitting: list[tuple[PythonRule, np.ndarray]] = []
        gates: list[tuple[PythonRule, np.ndarray]] = []
        seen: set[bytes] = set()
        known = dict(zip(primitives, self._condition_evaluator.masks(domain, rows, primitives), strict=True))
        for primitive, mask in known.items():
            if mask is None or not mask.any() or mask.all():
                continue
            if mask.tobytes() in seen or (~mask).tobytes() in seen:
                continue
            seen.add(mask.tobytes())
            matches = np.bincount(arrays.states, weights=mask.astype(float), minlength=len(counts))
            if np.all((matches == 0) | (matches == counts)):
                gates.append((primitive, mask))
            else:
                splitting.append((primitive, mask))

        masks = dict((*splitting, *gates))
        everything = np.ones(len(rows), dtype=bool)
        judge = _Judge(action, arrays, self._advantage_contrast, settings)
        found: list[tuple[Hypothesis, np.ndarray]] = []
        level = self._best(
            [
                (hypothesis, mask, index)
                for index, (primitive, mask) in enumerate(splitting)
                if (hypothesis := judge.hypothesis((), (primitive,), everything, mask)) is not None
            ],
            settings.beam_width,
        )
        found.extend((hypothesis, mask) for hypothesis, mask, _ in level)
        frontier = [(hypothesis.conditions, mask, index) for hypothesis, mask, index in level]
        frontier.extend(((gate,), mask, -1) for gate, mask in gates if arrays.visits[mask].sum() >= settings.min_rule_visits)
        skipped = 0
        for _ in range(2, settings.max_conditions + 1):
            candidates = []
            for conditions, scope, last in frontier:
                for index in range(last + 1, len(splitting)):
                    primitive, mask = splitting[index]
                    child = scope & mask
                    if not child.any():
                        continue
                    extended = (*conditions, primitive)
                    if self._reducible(extended, child, masks, everything):
                        skipped += 1
                        continue
                    hypothesis = judge.hypothesis(conditions, extended, scope, child)
                    if hypothesis is not None:
                        candidates.append((hypothesis, child, index))
            level = self._best(candidates, settings.beam_width)
            found.extend((hypothesis, child) for hypothesis, child, _ in level)
            frontier = [(hypothesis.conditions, child, index) for hypothesis, child, index in level]

        shortened = 0
        own_patterns = [pattern for pattern in patterns if pattern.action == action]
        missing = list(
            dict.fromkeys(
                condition for pattern in own_patterns for condition in pattern.conditions if condition not in known
            )
        )
        known.update(zip(missing, self._condition_evaluator.masks(domain, rows, missing), strict=True))
        for pattern in own_patterns:
            pattern_masks = [known[condition] for condition in pattern.conditions]
            if any(mask is None for mask in pattern_masks):
                continue
            conditions, matching = self._shortened(pattern.conditions, pattern_masks, everything)  # type: ignore[arg-type]
            shortened += len(conditions) < len(pattern.conditions)
            if not conditions:
                continue
            hypothesis = judge.hypothesis((), conditions, everything, matching)
            if hypothesis is not None:
                found.append((hypothesis, matching))

        merged = self._merged(found)
        logger.info(
            "%s: skipped %d combinations a condition adds nothing to, shortened %d goal patterns, merged %d hypotheses "
            "matching the same rows",
            action,
            skipped,
            shortened,
            len(found) - len(merged),
        )
        return merged

    def _best(
        self, candidates: list[tuple[Hypothesis, np.ndarray, int]], width: int
    ) -> list[tuple[Hypothesis, np.ndarray, int]]:
        return sorted(candidates, key=lambda candidate: -candidate[0].score)[:width]

    def _reducible(
        self,
        conditions: tuple[PythonRule, ...],
        matching: np.ndarray,
        masks: Mapping[PythonRule, np.ndarray],
        everything: np.ndarray,
    ) -> bool:
        """Whether one of the conditions can be dropped without changing the rows they match. Dropping one at a time is
        enough: dropping more can only match more rows."""
        return any(
            np.array_equal(
                np.logical_and.reduce([everything, *(masks[other] for other in conditions[:skip] + conditions[skip + 1 :])]),
                matching,
            )
            for skip in range(len(conditions))
        )

    def _shortened(
        self, conditions: tuple[PythonRule, ...], masks: list[np.ndarray], everything: np.ndarray
    ) -> tuple[tuple[PythonRule, ...], np.ndarray]:
        """The conditions left after dropping them, first to last, while the rows they match stay the same, and those
        rows."""
        matching = np.logical_and.reduce([everything, *masks])
        kept = list(range(len(conditions)))
        for index in range(len(conditions)):
            others = [masks[other] for other in kept if other != index]
            if np.array_equal(np.logical_and.reduce([everything, *others]), matching):
                kept.remove(index)
        return tuple(conditions[index] for index in kept), matching

    def _merged(self, found: list[tuple[Hypothesis, np.ndarray]]) -> tuple[Hypothesis, ...]:
        """One hypothesis per set of rows matched: the fewest conditions, then the highest score, then the first found;
        in the order found."""
        best: dict[bytes, Hypothesis] = {}
        for hypothesis, matching in found:
            key = matching.tobytes()
            current = best.get(key)
            if current is None or (len(hypothesis.conditions), -hypothesis.score) < (
                len(current.conditions),
                -current.score,
            ):
                best[key] = hypothesis
        chosen = {id(hypothesis) for hypothesis in best.values()}
        return tuple(hypothesis for hypothesis, _ in found if id(hypothesis) in chosen)


class _Judge:
    """Turns a comparison into a hypothesis when it passes discovery's thresholds."""

    def __init__(
        self, action: str, arrays: RowArrays, advantage_contrast: AdvantageContrast, settings: GenerationSettings
    ) -> None:
        self._action = action
        self._arrays = arrays
        self._advantage_contrast = advantage_contrast
        self._settings = settings
        self._low = float(arrays.payoffs.min())
        self._high = float(arrays.payoffs.max())
        self._z = NormalDist().inv_cdf(0.5 + settings.confidence / 2)

    def hypothesis(
        self,
        parent: tuple[PythonRule, ...],
        conditions: tuple[PythonRule, ...],
        scope: np.ndarray,
        matching: np.ndarray,
    ) -> Hypothesis | None:
        arrays, settings = self._arrays, self._settings
        inside = scope & matching
        visits = float(arrays.visits[inside].sum())
        if visits < settings.min_rule_visits:
            return None
        differences = self._advantage_contrast.differences(arrays, scope, inside)
        if len(differences) < MIN_STATES:
            return None
        effect = float(differences.mean())
        if abs(effect) < settings.min_gain:
            return None
        deviation = float(differences.std(ddof=1))
        score = abs(effect) / (deviation / math.sqrt(len(differences))) if deviation > 0 else math.inf
        expected_value, error = self._payoff(inside)
        spread = self._high - self._low
        priority = spread > 0 and (
            expected_value - self._z * error >= self._high - PRIORITY_MARGIN * spread
            or expected_value + self._z * error <= self._low + PRIORITY_MARGIN * spread
        )
        return Hypothesis(
            self._action,
            conditions,
            parent,
            1 if effect > 0 else -1,
            effect,
            len(differences),
            int(visits),
            expected_value,
            priority,
            score,
        )

    def _payoff(self, mask: np.ndarray) -> tuple[float, float]:
        """The visit-weighted mean payoff and its standard error, shrunk toward the widest spread the payoffs allow."""
        weights, payoffs = self._arrays.visits[mask], self._arrays.payoffs[mask]
        total = float(weights.sum())
        mean = float((weights * payoffs).sum() / total)
        variance = float((weights * (payoffs - mean) ** 2).sum() / total)
        effective = total**2 / float((weights**2).sum())
        prior = ((self._high - self._low) / 2) ** 2
        shrunk = (variance * effective + prior) / (effective + 1)
        return mean, math.sqrt(shrunk / effective)
