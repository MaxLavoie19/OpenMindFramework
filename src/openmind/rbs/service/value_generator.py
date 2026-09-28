import logging
import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np

from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.inference.constant.inference_constant import SINGLE_TARGET
from openmind.knowledge.constant.knowledge_constant import INFERENCE

from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.source import Source
from openmind.model.service.model_registry import ModelRegistry
from openmind.inference.model.expression import Expression
from openmind.inference.model.expression_search_result import ExpressionSearchResult
from openmind.inference.model.search_budget import SearchBudget
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_search import ExpressionSearch
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.sparse_fit import SparseFit
from openmind.rbs.model.value_fit import ValueFit
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.model.value_generation_result import ValueGenerationResult
from openmind.rbs.model.heuristic_target import HeuristicTarget
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.sparse_fitter import SparseFitter

logger = logging.getLogger(__name__)

#: The position rule standing for what a fit leaves when every term reads nothing: a constant, weighted by the bias.
CONSTANT_RULE = "a position is worth this much before anything is read"
CONSTANT_SOURCE = "1.0"

#: What a ruleset holding one price's fit is called, where every price is kept. The fit the held-out rows chose
#: keeps the task's own name, so whatever asked for the position value before still gets the one that was chosen.
PRICED_RULESET = "{task} at price {price}"

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
        rbs: RuleBasedGame,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        settings: ValueSettings,
        target: HeuristicTarget,
        seeds: Sequence[Expression | tuple[Expression, float]] = (),
        dropped: Collection[str] = (),
    ) -> ValueGenerationResult:
        """Without held-out rows, the fit with the lowest training loss is kept; ties go to the fewest terms. The search
        runs at the middle price of the sweep, trying the seeds first.

        A seed may carry the weight the rules imply it should start from; see `ExpressionSearch.search`."""
        payoffs = np.array([row.target for row in training], dtype=float)
        held_out_payoffs = np.array([row.target for row in held_out], dtype=float)
        return self.generate_for_targets(
            rbs, training, held_out, {SINGLE_TARGET: (payoffs, held_out_payoffs)}, settings, target, seeds, dropped
        )[
            SINGLE_TARGET
        ]

    def generate_for_targets(
        self,
        rbs: RuleBasedGame,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        targets: Mapping[str, TargetValues],
        settings: ValueSettings,
        target: HeuristicTarget,
        seeds: Sequence[Expression | tuple[Expression, float]] = (),
        dropped: Collection[str] = (),
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
                results[name] = ValueGenerationResult(target.context, (), (), None, ())
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
                dropped,
            )
            for name, (values, held_out_values) in scaled.items():
                results[name] = self._sweep(
                    rbs, target, self._label(name, several), found, values, held_out_values, prices, settings, bool(held_out)
                )
        return {name: results[name] for name in targets}

    def _sweep(
        self,
        rbs: RuleBasedGame,
        target: HeuristicTarget,
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
        # How far each price's loss could be out, being a mean over so many rows. What it is for is below.
        widths: list[float] = []
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
            width = (
                self._sparse_fitter.uncertainty(held_standard, held_out_targets, fit.weights, fit.bias)
                if has_held_out
                else self._sparse_fitter.uncertainty(standard, targets, fit.weights, fit.bias)
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
            widths.append(width)

        index = self._chosen(fits, widths, label)
        chosen = fitted[index]
        kept = sorted((at for at, weight in enumerate(chosen.weights) if weight != 0.0), key=lambda at: -abs(chosen.weights[at]))
        declared = self._declared(target, target.ruleset, chosen, terms, means, scales, label, fits[index].price)
        others: list[tuple[str, tuple[RuleRecord, ...]]] = []
        if settings.keep_every_price:
            for at, fit in enumerate(fitted):
                if at == index:
                    continue
                named = PRICED_RULESET.format(task=target.ruleset, price=f"{fits[at].price:g}")
                others.append((named, self._declared(target, named, fit, terms, means, scales, label, fits[at].price)))
            logger.info(
                "%sKept every price as a heuristic of its own to be played: %s",
                label,
                "; ".join(f"{named} with {len(rules) - 1} rules" for named, rules in others) or "there was only one",
            )
        strengths = tuple((terms[at], float(chosen.weights[at])) for at in kept)
        return ValueGenerationResult(
            target.context, declared, tuple(fits), fits[index], terms, strengths, tuple(others)
        )

    def _declared(
        self,
        target: HeuristicTarget,
        ruleset_name: str,
        fit: SparseFit,
        terms: Sequence[PythonRule],
        means: Sequence[float],
        scales: Sequence[float],
        label: str,
        price: float,
    ) -> tuple[RuleRecord, ...]:
        """One fit written into a ruleset of its own: its non-zero terms at their weights on the values as read,
        and the constant left when every term reads nothing."""
        kept = sorted((at for at, weight in enumerate(fit.weights) if weight != 0.0), key=lambda at: -abs(fit.weights[at]))
        bias = fit.bias - math.fsum(fit.weights[at] * float(means[at]) / float(scales[at]) for at in kept)
        rules = [self._link(target, ruleset_name, CONSTANT_RULE, PythonRule(CONSTANT_SOURCE), bias)]
        for at in kept:
            weight = fit.weights[at] / float(scales[at])
            rules.append(self._link(target, ruleset_name, terms[at].source, terms[at], weight))
            logger.debug("%s%s: %+.6g × %s", label, ruleset_name, weight, terms[at].source)
        logger.info(
            "%sPrice %s into %s: %d position rules and a constant of %s",
            label,
            price,
            ruleset_name,
            len(rules) - 1,
            bias,
        )
        return tuple(rules)

    def _chosen(self, fits: Sequence[ValueFit], widths: Sequence[float], label: str) -> int:
        """Which price to keep: the fewest terms among the fits nothing told apart from the best.

        **A loss is a mean over rows and a mean has a width.** Preferring the lowest loss outright made a fit
        of three thousand terms beat one of forty by two ten-thousandths, measured over twenty rows, every
        time — a difference in the fifth significant figure deciding a difference of eighty times in size. The
        tie-break on fewest terms was there and never fired, because it waited for two floats to be exactly
        equal and floats never are.

        So near enough counts as equal: a fit whose loss is within the best fit's standard error has not been
        told apart from it by the rows it was measured on, and among those the smallest is kept. The width is
        derived from the rows rather than chosen, which is the only reason to trust it — nobody picked a
        tolerance, the evidence said how much it could resolve."""
        best = min(
            range(len(fits)),
            key=lambda at: (fits[at].training_loss if fits[at].held_out_loss is None else fits[at].held_out_loss),
        )
        lowest = fits[best].training_loss if fits[best].held_out_loss is None else fits[best].held_out_loss
        within = lowest + widths[best]
        alike = [
            at
            for at in range(len(fits))
            if (fits[at].training_loss if fits[at].held_out_loss is None else fits[at].held_out_loss) <= within
        ]
        chosen = min(alike, key=lambda at: (fits[at].terms_kept, at))
        if chosen != best:
            logger.info(
                "%sPrice %s keeps %d terms where price %s keeps %d, and %g of loss does not tell them apart "
                "(within %g, what %d rows can resolve)",
                label,
                fits[chosen].price,
                fits[chosen].terms_kept,
                fits[best].price,
                fits[best].terms_kept,
                abs((fits[chosen].held_out_loss or fits[chosen].training_loss) - lowest),
                widths[best],
                len(fits),
            )
        return chosen

    def _link(
        self, target: HeuristicTarget, ruleset_name: str, name: str, rule: PythonRule, weight: float
    ) -> RuleRecord:
        """Declares a fitted position rule, open, and links it into that ruleset of the target's context at its
        weight; a rule of the same name already there is revised in place and its weight set anew.

        **What task it is a model of and what kind its rules are come from the target.** They were constants
        here, so a fit over what each action is worth would have been declared a position value and linked into
        a position value's ruleset, where anything asking for a position value would have found it and believed
        it. Nothing about fitting cares which of the two it is; only the naming did.

        **The ruleset is named because a context may hold several.** One model of a task was one ruleset while
        fitting kept one fit; several fits of the same task are several models, which is what the registry is
        for. The task each is a model of stays `position value`, so whatever asks for models of that task finds
        all of them and can tell them apart by playing."""
        knowledge_base = target.knowledge_base
        context_id = knowledge_base.ensure_context(target.context).id
        mechanism = knowledge_base.ensure_mechanism(INFERENCE).id
        ruleset = knowledge_base.ruleset_named(context_id, ruleset_name)
        if ruleset is None:
            ruleset = knowledge_base.ruleset(
                Ruleset(ruleset_name, context_id, target.task, Source(mechanism, (("method", "fit"),)), open=True)
            )
            ModelRegistry(AccuracyScorer()).register_ruleset(knowledge_base, ruleset)
        standing = next((held for held, _ in knowledge_base.ruleset_rules(ruleset.id, (target.kind,)) if held.name == name), None)
        declared = knowledge_base.declare(
            RuleRecord(
                name,
                target.kind,
                rule,
                Source(mechanism, (("method", "fit"), ("context", target.context))),
                id="" if standing is None else standing.id,
            )
        )
        knowledge_base.link(ruleset.id, declared.id, weight)
        return declared

    def _label(self, name: str, several: bool) -> str:
        return f"{name}: " if several else ""
