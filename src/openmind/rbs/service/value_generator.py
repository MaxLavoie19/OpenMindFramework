import logging
import math
from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.inference.model.expression import Expression
from openmind.inference.model.search_budget import SearchBudget
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_search import ExpressionSearch
from openmind.rbs.mapper.value_rule_text_mapper import ValueRuleTextMapper
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.sparse_fit import SparseFit
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_fit import ValueFit
from openmind.rbs.model.value_generation_result import ValueGenerationResult
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.sparse_fitter import SparseFitter

logger = logging.getLogger(__name__)


class ValueGenerator:
    """Generates value rules for any domain from positions and the payoffs they led to: searches expressions of the
    positions and of what the domain's own actions make of them, within the settings' time and memory budget, then fits
    the expressions' weights, each priced per clause, at each price of a sweep, keeping the fit whose rules predict the
    held-out positions best."""

    def __init__(
        self,
        expression_search: ExpressionSearch,
        expression_generator: ExpressionGenerator,
        sparse_fitter: SparseFitter,
        value_rule_text_mapper: ValueRuleTextMapper,
    ) -> None:
        self._expression_search = expression_search
        self._expression_generator = expression_generator
        self._sparse_fitter = sparse_fitter
        self._value_rule_text_mapper = value_rule_text_mapper

    def generate(
        self,
        domain: Domain,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        settings: ValueSettings,
        seeds: Sequence[Expression] = (),
    ) -> ValueGenerationResult:
        """Without held-out rows, the fit with the lowest training loss is kept; ties go to the fewest terms. The search
        runs at the middle price of the sweep, trying the seeds first."""
        if not training:
            raise ValueError("Value generation needs training rows")
        if not settings.prices:
            raise ValueError("Value generation needs at least one price")
        payoffs = np.array([row.target for row in training], dtype=float)
        low, high = float(payoffs.min()), float(payoffs.max())
        if high == low:
            logger.info("Every training payoff is %s: nothing to fit", low)
            return ValueGenerationResult(ValueBase(domain.name, 0.0, low, high, ()), (), None, ())

        targets = (payoffs - low) / (high - low)
        held_out_targets = (np.array([row.target for row in held_out], dtype=float) - low) / (high - low)
        prices = sorted(settings.prices, reverse=True)
        found = self._expression_search.search(
            domain,
            training,
            held_out,
            targets,
            prices[len(prices) // 2],
            settings.max_steps,
            settings.tolerance,
            SearchBudget(settings.seconds, settings.memory_bytes, settings.candidates),
            seeds,
        )
        expressions = found.expressions
        terms = tuple(self._expression_generator.source(expression) for expression in expressions)
        logger.info(
            "%d candidate terms after %d generations of search (%s), looking up to %d actions ahead",
            len(terms),
            found.generations,
            found.stopped,
            max((expression.plies for expression in expressions), default=0),
        )

        costs = np.array([expression.clauses for expression in expressions], dtype=float)
        matrix = np.column_stack(found.training) if terms else np.empty((len(training), 0))
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        standard = (matrix - means) / scales
        held_matrix = np.column_stack(found.held_out) if terms else np.empty((len(held_out), 0))
        held_standard = (held_matrix - means) / scales

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
                self._sparse_fitter.loss(held_standard, held_out_targets, fit.weights, fit.bias) if held_out else None,
            )
            logger.info(
                "Price %s: %d of %d terms kept in %d steps, %s; training loss %s, held-out loss %s",
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
        rules = tuple(ValueRule(terms[at], chosen.weights[at] / float(scales[at])) for at in kept)
        bias = chosen.bias - math.fsum(chosen.weights[at] * float(means[at]) / float(scales[at]) for at in kept)
        logger.info(
            "Chose price %s: %d value rules, bias %s, payoffs from %s to %s",
            fits[index].price,
            len(rules),
            bias,
            low,
            high,
        )
        for rule in rules:
            logger.debug("%s", self._value_rule_text_mapper.to_text(rule))
        return ValueGenerationResult(ValueBase(domain.name, bias, low, high, rules), tuple(fits), fits[index], terms)
