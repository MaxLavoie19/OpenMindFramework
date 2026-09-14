import itertools
import logging
import math
from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.mapper.value_rule_text_mapper import ValueRuleTextMapper
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.sparse_fit import SparseFit
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_fit import ValueFit
from openmind.rbs.model.value_generation_result import ValueGenerationResult
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.sparse_fitter import SparseFitter
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rbs.service.term_generator import TermGenerator
from openmind.rule.model.python_rule import PythonRule

logger = logging.getLogger(__name__)

type Columns = list[np.ndarray]


class ValueGenerator:
    """Generates value rules for any domain from positions and the payoffs they led to: generates single terms, adds the
    products of every pair among the single terms most correlated with the payoffs, and fits the terms' weights with a
    price on them at each price of a sweep, keeping the fit whose rules predict the held-out positions best."""

    def __init__(
        self,
        term_generator: TermGenerator,
        term_evaluator: TermEvaluator,
        sparse_fitter: SparseFitter,
        value_rule_text_mapper: ValueRuleTextMapper,
    ) -> None:
        self._term_generator = term_generator
        self._term_evaluator = term_evaluator
        self._sparse_fitter = sparse_fitter
        self._value_rule_text_mapper = value_rule_text_mapper

    def generate(
        self,
        domain: Domain,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        settings: ValueSettings,
    ) -> ValueGenerationResult:
        """Without held-out rows, the fit with the lowest training loss is kept; ties go to the fewest terms."""
        if not training:
            raise ValueError("Value generation needs training rows")
        if not settings.prices:
            raise ValueError("Value generation needs at least one price")
        payoffs = np.array([row.target for row in training], dtype=float)
        low, high = float(payoffs.min()), float(payoffs.max())
        singles = self._term_generator.generate(domain, training, settings)
        terms, train, test = self._usable(
            singles,
            self._term_evaluator.columns(domain, training, singles),
            self._term_evaluator.columns(domain, held_out, singles),
        )
        logger.info(
            "%d single terms generated, %d usable on %d training and %d held-out rows",
            len(singles),
            len(terms),
            len(training),
            len(held_out),
        )
        if high == low:
            logger.info("Every training payoff is %s: nothing to fit", low)
            return ValueGenerationResult(ValueBase(domain.name, 0.0, low, high, ()), (), None, tuple(terms))

        targets = (payoffs - low) / (high - low)
        held_out_targets = (np.array([row.target for row in held_out], dtype=float) - low) / (high - low)
        pairs, pair_train, pair_test = self._pairs(terms, train, test, targets, settings.pair_pool)
        singles_count = len(terms)
        terms, train, test = self._usable([*terms, *pairs], [*train, *pair_train], [*test, *pair_test])
        logger.info(
            "%d candidate terms: %d single, %d products of pairs among the %d single terms most correlated with the payoffs",
            len(terms),
            singles_count,
            len(terms) - singles_count,
            min(settings.pair_pool, singles_count),
        )

        matrix = np.column_stack(train) if train else np.empty((len(training), 0))
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        standard = (matrix - means) / scales
        held_matrix = np.column_stack(test) if test else np.empty((len(held_out), 0))
        held_standard = (held_matrix - means) / scales

        fits: list[ValueFit] = []
        fitted: list[SparseFit] = []
        start: SparseFit | None = None
        for price in sorted(settings.prices, reverse=True):
            fit = self._sparse_fitter.fit(standard, targets, price, settings.max_steps, settings.tolerance, start)
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
        return ValueGenerationResult(ValueBase(domain.name, bias, low, high, rules), tuple(fits), fits[index], tuple(terms))

    def _usable(
        self,
        terms: Sequence[PythonRule],
        train: Sequence[np.ndarray | None],
        test: Sequence[np.ndarray | None],
    ) -> tuple[list[PythonRule], Columns, Columns]:
        """The terms that give a number on every row, vary on the training rows, and don't repeat an earlier term's
        training values."""
        kept: tuple[list[PythonRule], Columns, Columns] = ([], [], [])
        seen: set[bytes] = set()
        for term, train_column, test_column in zip(terms, train, test, strict=True):
            if train_column is None or test_column is None or np.ptp(train_column) == 0.0:
                continue
            key = train_column.tobytes()
            if key in seen:
                continue
            seen.add(key)
            kept[0].append(term)
            kept[1].append(train_column)
            kept[2].append(test_column)
        return kept

    def _pairs(
        self, terms: Sequence[PythonRule], train: Columns, test: Columns, targets: np.ndarray, pool: int
    ) -> tuple[list[PythonRule], Columns, Columns]:
        """The product of every pair among the pool single terms most correlated with the targets, in generation order."""
        centered = targets - targets.mean()

        def correlation(column: np.ndarray) -> float:
            deviation = column - column.mean()
            denominator = math.sqrt(float(deviation @ deviation) * float(centered @ centered))
            return abs(float(deviation @ centered)) / denominator if denominator else 0.0

        best = sorted(sorted(range(len(terms)), key=lambda at: -correlation(train[at]))[:pool])
        pairs: tuple[list[PythonRule], Columns, Columns] = ([], [], [])
        for first, second in itertools.combinations(best, 2):
            pairs[0].append(PythonRule(f"({terms[first].source}) * ({terms[second].source})"))
            pairs[1].append(train[first] * train[second])
            pairs[2].append(test[first] * test[second])
        return pairs
