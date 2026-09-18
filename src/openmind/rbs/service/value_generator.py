import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np

from openmind.inference.constant.inference_constant import SINGLE_TARGET
from openmind.inference.model.expression import Expression
from openmind.inference.model.expression_search_result import ExpressionSearchResult
from openmind.inference.model.search_budget import SearchBudget
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_search import ExpressionSearch
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.sparse_fit import SparseFit
from openmind.rbs.model.value_fit import ValueFit
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.model.value_generation_result import ValueGenerationResult
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.sparse_fitter import SparseFitter

logger = logging.getLogger(__name__)

#: The position rule standing for what a fit leaves when every term reads nothing: a constant, weighted by the bias.
CONSTANT_RULE = "a position is worth this much before anything is read"
CONSTANT_SOURCE = "1.0"

#: A target's values on the training rows and on the held-out rows, in the rows' order.
type TargetValues = tuple[np.ndarray, np.ndarray]


class ValueGenerator:
    """Generates value rules for any domain from positions and the payoffs they led to: searches expressions of the
    positions and of what the domain's own actions make of them, within the settings' time and memory budget, then fits
    the expressions' weights, each priced per clause and per share of rows where it isn't blank, at each price of a sweep,
    keeping the fit whose rules predict the held-out positions best. A blank reads as 0, and a term with blanks isn't
    centered, so a rule adds nothing where its term gives None. Given several targets,
    one search keeps what any target supports, and each target gets its own sweep and value base."""

    def __init__(
        self,
        expression_search: ExpressionSearch,
        expression_generator: ExpressionGenerator,
        sparse_fitter: SparseFitter,
    ) -> None:
        self._expression_search = expression_search
        self._expression_generator = expression_generator
        self._sparse_fitter = sparse_fitter

    def generate(
        self,
        rbs: RuleBasedSystem,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        settings: ValueSettings,
        declarer: RuleDeclarer,
        seeds: Sequence[Expression] = (),
    ) -> ValueGenerationResult:
        """Without held-out rows, the fit with the lowest training loss is kept; ties go to the fewest terms. The search
        runs at the middle price of the sweep, trying the seeds first."""
        payoffs = np.array([row.target for row in training], dtype=float)
        held_out_payoffs = np.array([row.target for row in held_out], dtype=float)
        return self.generate_for_targets(
            rbs, training, held_out, {SINGLE_TARGET: (payoffs, held_out_payoffs)}, settings, declarer, seeds
        )[
            SINGLE_TARGET
        ]

    def generate_for_targets(
        self,
        rbs: RuleBasedSystem,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        targets: Mapping[str, TargetValues],
        settings: ValueSettings,
        declarer: RuleDeclarer,
        seeds: Sequence[Expression] = (),
    ) -> dict[str, ValueGenerationResult]:
        """Each target's result, in the targets' order. A target is its values on the training and held-out rows, scaled
        from its lowest to its highest training value; one that never varies has nothing to fit and isn't searched. No
        training row or no price raises ValueError."""
        if not training:
            raise ValueError("Value generation needs training rows")
        if not settings.prices:
            raise ValueError("Value generation needs at least one price")
        several = len(targets) > 1
        results: dict[str, ValueGenerationResult] = {}
        scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (values, held_out_values) in targets.items():
            if float(np.min(values)) == float(np.max(values)):
                logger.info(
                    "%sEvery training payoff is %s: nothing to fit", self._label(name, several), float(np.min(values))
                )
                results[name] = ValueGenerationResult(declarer.context, (), (), None, ())
                continue
            scaled[name] = (values, held_out_values)
        if scaled:
            prices = sorted(settings.prices, reverse=True)
            found = self._expression_search.search(
                rbs,
                training,
                held_out,
                {name: values for name, (values, _) in scaled.items()},
                prices[len(prices) // 2],
                settings.max_steps,
                settings.tolerance,
                SearchBudget(settings.seconds, settings.memory_bytes, settings.candidates),
                seeds,
            )
            for name, (values, held_out_values) in scaled.items():
                results[name] = self._sweep(
                    rbs, declarer, self._label(name, several), found, values, held_out_values, prices, settings, bool(held_out)
                )
        return {name: results[name] for name in targets}

    def _sweep(
        self,
        rbs: RuleBasedSystem,
        declarer: RuleDeclarer,
        label: str,
        found: ExpressionSearchResult,
        targets: np.ndarray,
        held_out_targets: np.ndarray,
        prices: Sequence[float],
        settings: ValueSettings,
        has_held_out: bool,
    ) -> ValueGenerationResult:
        """One target's fits at every price over the search's expressions, and the value base of the fit chosen."""
        expressions = found.expressions
        terms = tuple(self._expression_generator.source(expression) for expression in expressions)
        logger.info(
            "%s%d candidate terms after %d generations of search (%s), looking up to %d actions ahead",
            label,
            len(terms),
            found.generations,
            found.stopped,
            max((expression.plies for expression in expressions), default=0),
        )
        rows, held_rows = len(targets), len(held_out_targets)
        search = self._expression_search
        costs = np.array(
            [expression.clauses * search.share(column) for expression, column in zip(expressions, found.training, strict=True)],
            dtype=float,
        )
        scalings = [search.scaling(column) for column in found.training]
        means = [center for center, _ in scalings]
        scales = [scale for _, scale in scalings]
        standard = (
            np.column_stack([search.standard(column, scaling) for column, scaling in zip(found.training, scalings, strict=True)])
            if terms
            else np.empty((rows, 0))
        )
        held_standard = (
            np.column_stack([search.standard(column, scaling) for column, scaling in zip(found.held_out, scalings, strict=True)])
            if terms
            else np.empty((held_rows, 0))
        )

        fits: list[ValueFit] = []
        fitted: list[SparseFit] = []
        start: SparseFit | None = None
        for price in prices:
            fit = self._sparse_fitter.fit(standard, targets, price, settings.max_steps, settings.tolerance, start, costs)
            start = fit
            value_fit = ValueFit(
                price,
                sum(1 for weight in fit.weights if weight != 0.0),
                fit.steps,
                fit.settled,
                self._sparse_fitter.loss(standard, targets, fit.weights, fit.bias),
                self._sparse_fitter.loss(held_standard, held_out_targets, fit.weights, fit.bias) if has_held_out else None,
            )
            logger.info(
                "%sPrice %s: %d of %d terms kept in %d steps, %s; training loss %s, held-out loss %s",
                label,
                price,
                value_fit.terms_kept,
                len(terms),
                value_fit.steps,
                "settled" if value_fit.settled else "not settled",
                value_fit.training_loss,
                value_fit.held_out_loss,
            )
            fits.append(value_fit)
            fitted.append(fit)

        index = min(
            range(len(fits)),
            key=lambda at: (
                fits[at].training_loss if fits[at].held_out_loss is None else fits[at].held_out_loss,
                fits[at].terms_kept,
            ),
        )
        chosen = fitted[index]
        kept = sorted((at for at, weight in enumerate(chosen.weights) if weight != 0.0), key=lambda at: -abs(chosen.weights[at]))
        bias = chosen.bias - math.fsum(chosen.weights[at] * float(means[at]) / float(scales[at]) for at in kept)
        declared = [declarer.position(CONSTANT_RULE, PythonRule(CONSTANT_SOURCE), bias)]
        for at in kept:
            weight = chosen.weights[at] / float(scales[at])
            declared.append(declarer.position(terms[at].source, terms[at], weight))
            logger.debug("%s%+.6g × %s", label, weight, terms[at].source)
        logger.info(
            "%sChose price %s: %d position rules and a constant of %s",
            label,
            fits[index].price,
            len(declared) - 1,
            bias,
        )
        strengths = tuple((terms[at], float(chosen.weights[at])) for at in kept)
        return ValueGenerationResult(declarer.context, tuple(declared), tuple(fits), fits[index], terms, strengths)

    def _label(self, name: str, several: bool) -> str:
        return f"{name}: " if several else ""
