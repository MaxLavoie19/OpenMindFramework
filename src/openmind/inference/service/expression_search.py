import gc
import hashlib
import itertools
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy.special import expit

from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import (
    CANDIDATE_BATCH,
    MIN_SCREENING_ROWS,
    SCREENING_SHARE,
    SINGLE_TARGET,
)
from openmind.inference.model.expression import Expression
from openmind.inference.model.expression_search_result import ExpressionSearchResult
from openmind.inference.model.search_budget import SearchBudget
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.parallel.model.call_over_memory import CallOverMemory
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.sparse_fitter import SparseFitter
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rule.model.python_rule import PythonRule

logger = logging.getLogger(__name__)

#: A column on the training rows and on the held-out rows.
type Columns = tuple[np.ndarray, np.ndarray]
#: A candidate: its expression; for one computed from kept columns, the operator, the columns it reads and the value of a
#: threshold; None for those of one evaluated as a rule.
type Candidate = tuple[Expression, str | None, Columns | None, Columns | None, float | None]
#: For every target by name: the kept expressions' weights at the price, the residual, and the training loss.
type Fits = dict[str, tuple[np.ndarray, np.ndarray, float]]


class ExpressionSearch:
    """Searches expressions that value positions, for any domain, from the leaves up, within a budget of time, memory and
    candidates. Each generation:

    1. fits the kept expressions at the price, a price per clause, and takes the residual of the scaled payoffs;
    2. expands every kept expression, the weighted first: its look-aheads, absolute value, thresholds, pattern and
       aggregate children once, and its combinations with every kept expression it hasn't met; then the expressions
       passed over since, the steepest first, into their look-aheads, pattern and aggregate children. Parents take turns,
       one candidate each, and candidates are made only as they are tried, never all at once;
    3. tries the candidates in batches: a candidate is evaluated on the screening rows and admitted only when its
       standardized gradient against the residual is above its price, the price times its clauses times the share of rows
       where it isn't blank, that is when the fit would give it a weight; then it is evaluated on every row and kept when
       it still is, varies, and repeats no kept column. Thresholds, absolute values and combinations are computed from the
       kept columns, a row blank in an operand staying blank; the others are evaluated as rules, in the evaluator's
       workers;
    4. evicts, when the kept columns and the fit's copies of them would pass the memory budget, the unweighted
       expressions with the smallest gradients first.

    Memory is measured, not estimated. Before each batch, when this process holds more than the memory budget, the search
    clears the views, forgets the expressions passed over and evicts every unweighted column; if the process still holds
    more, it stops. Every process evaluating terms clears its views once it holds more than its share of the budget.

    It stops when the time, memory or candidate budget runs out, or when a generation has nothing left to try. A
    generation that keeps nothing doesn't stop it: the next one builds on the expressions it passed over.

    A blank is a row where a term gives None: what it reads isn't there at that moment, as a fork detector without a
    fork. A column with blanks is scaled without centering, its blanks reading as 0, so a blank adds nothing to the fit,
    and a rare term is priced for the rows where it speaks rather than for every row.

    Given several targets, such as the targets of several signals, every generation fits each target, and a candidate is
    admitted when its gradient against any target's residual is above its price: an expression is kept when any target
    supports it. Expansion and eviction then go by the largest weight and gradient over the targets."""

    def __init__(
        self,
        expression_generator: ExpressionGenerator,
        term_evaluator: TermEvaluator,
        sparse_fitter: SparseFitter,
        memory_meter: MemoryMeter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._expression_generator = expression_generator
        self._term_evaluator = term_evaluator
        self._sparse_fitter = sparse_fitter
        self._memory_meter = memory_meter
        self._clock = clock

    def search(
        self,
        domain: Domain,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        targets: np.ndarray | Mapping[str, np.ndarray],
        price: float,
        max_steps: int,
        tolerance: float,
        budget: SearchBudget,
        seeds: Sequence[Expression] = (),
    ) -> ExpressionSearchResult:
        """Targets are the training rows' payoffs scaled from 0 to 1, or several such targets by name. Seeds, such as the
        expressions a deduction induced, are tried in the first generation before the leaves, in their order. No target
        raises ValueError."""
        named = (
            {SINGLE_TARGET: np.asarray(targets, dtype=float)}
            if isinstance(targets, np.ndarray)
            else {name: np.asarray(values, dtype=float) for name, values in targets.items()}
        )
        if not named:
            raise ValueError("An expression search needs at least one target")
        deadline = self._clock() + budget.seconds
        generator = self._expression_generator
        self._term_evaluator.limit_memory(budget.memory_bytes)
        vocabulary = generator.vocabulary(domain, (row.state for row in training))
        screening = self._screening(len(training))
        screen_rows = [training[index] for index in screening]
        capacity = max(1, budget.memory_bytes // (3 * 8 * max(1, len(training) + len(held_out))))
        expressions: list[Expression] = []
        kept: list[Columns] = []
        keys: set[bytes] = set()
        tried: set[bytes] = set()
        expanded: set[bytes] = set()
        combined: dict[bytes, set[bytes]] = {}
        passed_over: list[tuple[Expression, float]] = []
        leaves = generator.leaves(vocabulary)
        first = tuple({expression.template: expression for expression in (*seeds, *leaves)}.values())
        logger.info(
            "Searching expressions for %s seconds within %d bytes, trying %s candidates: %d seeds, %d leaves, %d training "
            "rows, %d screened",
            budget.seconds,
            budget.memory_bytes,
            "any number of" if budget.candidates is None else f"at most {budget.candidates}",
            len(seeds),
            len(leaves),
            len(training),
            len(screen_rows),
        )
        generation, stopped, tried_total = 0, "", 0
        while not stopped:
            generation += 1
            fits = self._fit_all(expressions, kept, named, price, max_steps, tolerance)
            residuals = {name: residual for name, (_, residual, _) in fits.items()}
            candidates: Iterator[Candidate] = (
                iter([(expression, None, None, None, None) for expression in first])
                if generation == 1
                else self._expand(expressions, kept, fits, vocabulary, expanded, combined, passed_over)
            )
            fresh = (candidate for candidate in candidates if self._digest(candidate[0].template) not in tried)
            tried_count = admitted = evicted = 0
            while True:
                if self._clock() >= deadline:
                    stopped = "the time budget ran out"
                    break
                room = CANDIDATE_BATCH if budget.candidates is None else min(CANDIDATE_BATCH, budget.candidates - tried_total)
                if room <= 0:
                    stopped = "the candidate budget ran out"
                    break
                used = self._memory_meter.resident_bytes()
                if used > budget.memory_bytes:
                    evicted += self._free(expressions, kept, keys, passed_over, used, budget, named, price, max_steps, tolerance)
                    if self._memory_meter.resident_bytes() > budget.memory_bytes:
                        stopped = "the memory budget ran out"
                        break
                batch = list(itertools.islice(fresh, room))
                if not batch:
                    break
                tried.update(self._digest(candidate[0].template) for candidate in batch)
                tried_count += len(batch)
                tried_total += len(batch)
                try:
                    admissions, passed = self._admitted(
                        domain, batch, training, held_out, screening, screen_rows, residuals, price
                    )
                except CallOverMemory as error:
                    logger.warning("Evaluating candidates took a worker over its memory cap twice: %s", error)
                    stopped = "the memory budget ran out"
                    break
                passed_over.extend(passed)
                for expression, columns in admissions:
                    key = columns[0].tobytes()
                    if key in keys:
                        continue
                    keys.add(key)
                    expressions.append(expression)
                    kept.append(columns)
                    admitted += 1
                    logger.debug("Kept %s", generator.source(expression).source)
                if len(expressions) > capacity:
                    evicted += self._evict(expressions, kept, keys, capacity, named, price, max_steps, tolerance)
            logger.info(
                "Generation %d: %d candidates tried, %d kept, %d evicted; %d expressions kept, %d weighted at price %s, "
                "looking up to %d actions ahead; training loss %s before the generation; %d candidates tried in all; "
                "%.0f seconds left; %d bytes held",
                generation,
                tried_count,
                admitted,
                evicted,
                len(expressions),
                int(np.count_nonzero(self._strongest(fits, len(expressions)))),
                price,
                max((expression.plies for expression in expressions), default=0),
                next(iter(fits.values()))[2] if len(fits) == 1 else " ".join(f"{name}={loss}" for name, (_, _, loss) in fits.items()),
                tried_total,
                max(0.0, deadline - self._clock()),
                self._memory_meter.resident_bytes(),
            )
            if not stopped and tried_count == 0:
                stopped = "nothing left to try"
        logger.info(
            "Search stopped after %d generations and %d candidates: %s; %d expressions kept",
            generation,
            tried_total,
            stopped,
            len(expressions),
        )
        return ExpressionSearchResult(
            tuple(expressions),
            tuple(columns[0] for columns in kept),
            tuple(columns[1] for columns in kept),
            generation,
            stopped,
            tried_total,
        )

    def _free(
        self,
        expressions: list[Expression],
        kept: list[Columns],
        keys: set[bytes],
        passed_over: list[tuple[Expression, float]],
        used: int,
        budget: SearchBudget,
        targets: Mapping[str, np.ndarray],
        price: float,
        max_steps: int,
        tolerance: float,
    ) -> int:
        """Clears the views, forgets the expressions passed over and evicts every column no target weights; the columns
        evicted."""
        self._term_evaluator.clear_memory()
        forgotten = len(passed_over)
        passed_over.clear()
        evicted = self._evict(expressions, kept, keys, None, targets, price, max_steps, tolerance) if kept else 0
        gc.collect()
        logger.info(
            "This process held %d bytes, over the memory budget of %d: cleared the views, forgot %d expressions passed "
            "over and evicted %d unweighted columns; it now holds %d bytes",
            used,
            budget.memory_bytes,
            forgotten,
            evicted,
            self._memory_meter.resident_bytes(),
        )
        return evicted

    def _admitted(
        self,
        domain: Domain,
        batch: Sequence[Candidate],
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        screening: np.ndarray,
        screen_rows: Sequence[PositionRow],
        residuals: Mapping[str, np.ndarray],
        price: float,
    ) -> tuple[list[tuple[Expression, Columns]], list[tuple[Expression, float]]]:
        """The candidates admitted, with their columns, and those that varied on the screening rows without being
        admitted, with their steepest gradient there; a gradient is the steepest over the targets' residuals."""
        residual = list(residuals.values())
        screen_residual = [values[screening] for values in residual]
        everything = len(screening) == len(training)
        found: list[tuple[Expression, Columns]] = []
        passed: list[tuple[Expression, float]] = []
        for expression, operator, first, second, value in batch:
            if operator is None or first is None:
                continue
            screened = self._compute(operator, first[0][screening], None if second is None else second[0][screening], value)
            gradient = self._gradient(screened, *screen_residual)
            if gradient <= self._price(price, expression, screened):
                if gradient > 0.0:
                    passed.append((expression, gradient))
                continue
            train = self._compute(operator, first[0], None if second is None else second[0], value)
            if self._usable(train) and self._gradient(train, *residual) > self._price(price, expression, train):
                found.append((expression, (train, self._compute(operator, first[1], None if second is None else second[1], value))))
            else:
                passed.append((expression, gradient))

        rules = [candidate[0] for candidate in batch if candidate[1] is None]
        if not rules:
            return found, passed
        sources = [self._expression_generator.source(expression) for expression in rules]
        screened_columns = self._term_evaluator.columns(domain, screen_rows, sources)
        passing: list[tuple[Expression, PythonRule, np.ndarray]] = []
        for expression, source, column in zip(rules, sources, screened_columns, strict=True):
            if column is None:
                continue
            gradient = self._gradient(column, *screen_residual)
            if gradient > self._price(price, expression, column):
                passing.append((expression, source, column))
            elif gradient > 0.0:
                passed.append((expression, gradient))
        if not passing:
            return found, passed
        passing_sources = [source for _, source, _ in passing]
        trains = (
            [column for _, _, column in passing]
            if everything
            else self._term_evaluator.columns(domain, training, passing_sources)
        )
        helds = self._term_evaluator.columns(domain, held_out, passing_sources)
        for (expression, _, column), train, held in zip(passing, trains, helds, strict=True):
            if train is not None and held is not None and self._usable(train):
                if self._gradient(train, *residual) > self._price(price, expression, train):
                    found.append((expression, (train, held)))
                    continue
            passed.append((expression, self._gradient(column, *screen_residual)))
        return found, passed

    def _expand(
        self,
        expressions: Sequence[Expression],
        kept: Sequence[Columns],
        fits: Fits,
        vocabulary: Vocabulary,
        expanded: set[bytes],
        combined: dict[bytes, set[bytes]],
        passed_over: list[tuple[Expression, float]],
    ) -> Iterator[Candidate]:
        """Every kept expression's candidates, the weighted first, by their largest weight then steepest gradient over
        the targets, then those of the expressions passed over since the last expansion, the steepest first; the parents
        take turns, and each makes its candidates only when asked."""
        residuals = [residual for _, residual, _ in fits.values()]
        gradients = [self._gradient(columns[0], *residuals) for columns in kept]
        strongest = self._strongest(fits, len(expressions))
        order = sorted(range(len(expressions)), key=lambda at: (strongest[at] == 0.0, -strongest[at], -gradients[at]))
        partners = [(expressions[at], kept[at]) for at in order]
        groups: list[Iterator[Candidate]] = [
            self._kept_candidates(expression, columns, partners, vocabulary, expanded, combined)
            for expression, columns in partners
        ]
        for expression, _ in sorted(passed_over, key=lambda item: -item[1]):
            digest = self._digest(expression.template)
            if digest not in expanded:
                expanded.add(digest)
                groups.append(self._passed_over_candidates(expression, vocabulary))
        passed_over.clear()
        return self._turns(groups)

    def _kept_candidates(
        self,
        expression: Expression,
        columns: Columns,
        partners: Sequence[tuple[Expression, Columns]],
        vocabulary: Vocabulary,
        expanded: set[bytes],
        combined: dict[bytes, set[bytes]],
    ) -> Iterator[Candidate]:
        generator = self._expression_generator
        digest = self._digest(expression.template)
        if digest not in expanded:
            expanded.add(digest)
            yield from ((child, None, None, None, None) for child in generator.look_aheads(expression))
            yield from ((child, operator, columns, None, None) for child, operator in generator.unary(expression))
            yield from (
                (child, relation, columns, None, cut)
                for child, relation, cut in generator.thresholds(expression, columns[0][~np.isnan(columns[0])].tolist())
            )
            yield from ((child, None, None, None, None) for child in generator.pattern_children(expression, vocabulary))
            yield from ((child, None, None, None, None) for child in generator.aggregate_children(expression, vocabulary))
        met = combined.setdefault(digest, set())
        for partner, partner_columns in partners:
            partner_digest = self._digest(partner.template)
            if partner_digest == digest or partner_digest in met:
                continue
            met.add(partner_digest)
            yield from (
                (child, operator, columns, partner_columns, None)
                for child, operator in generator.combinations(expression, partner)
            )

    def _passed_over_candidates(self, expression: Expression, vocabulary: Vocabulary) -> Iterator[Candidate]:
        generator = self._expression_generator
        yield from ((child, None, None, None, None) for child in generator.look_aheads(expression))
        yield from ((child, None, None, None, None) for child in generator.pattern_children(expression, vocabulary))
        yield from ((child, None, None, None, None) for child in generator.aggregate_children(expression, vocabulary))

    def _turns(self, groups: Iterable[Iterator[Candidate]]) -> Iterator[Candidate]:
        """One candidate from each group in turn, until every group is done."""
        live = deque(groups)
        while live:
            group = live.popleft()
            candidate = next(group, None)
            if candidate is not None:
                live.append(group)
                yield candidate

    def _fit_all(
        self,
        expressions: Sequence[Expression],
        kept: Sequence[Columns],
        targets: Mapping[str, np.ndarray],
        price: float,
        max_steps: int,
        tolerance: float,
    ) -> Fits:
        return {name: self._fit(expressions, kept, values, price, max_steps, tolerance) for name, values in targets.items()}

    def _strongest(self, fits: Fits, count: int) -> np.ndarray:
        """Each kept expression's largest weight over the targets, in size."""
        if count == 0:
            return np.zeros(0)
        return np.max(np.abs(np.vstack([weights for weights, _, _ in fits.values()])), axis=0)

    def _fit(
        self,
        expressions: Sequence[Expression],
        kept: Sequence[Columns],
        targets: np.ndarray,
        price: float,
        max_steps: int,
        tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """The weights at the price, the residual of the scaled payoffs (prediction minus payoff), and the training
        loss."""
        if not kept:
            mean = float(np.mean(targets))
            prediction = np.full(len(targets), mean)
            clipped = min(max(mean, 1e-12), 1.0 - 1e-12)
            loss = float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)))
            return np.zeros(0), prediction - targets, loss
        standard = np.column_stack([self.standard(columns[0]) for columns in kept])
        costs = np.array(
            [expression.clauses * self.share(columns[0]) for expression, columns in zip(expressions, kept, strict=True)],
            dtype=float,
        )
        fit = self._sparse_fitter.fit(standard, targets, price, max_steps, tolerance, None, costs)
        weights = np.asarray(fit.weights, dtype=float)
        prediction = expit(standard @ weights + fit.bias)
        return weights, prediction - targets, self._sparse_fitter.loss(standard, targets, fit.weights, fit.bias)

    def _evict(
        self,
        expressions: list[Expression],
        kept: list[Columns],
        keys: set[bytes],
        capacity: int | None,
        targets: Mapping[str, np.ndarray],
        price: float,
        max_steps: int,
        tolerance: float,
    ) -> int:
        """Keeps at most capacity expressions, or only those some target weights without a capacity, dropping those no
        target weights with the smallest gradients first; the number dropped."""
        fits = self._fit_all(expressions, kept, targets, price, max_steps, tolerance)
        strongest = self._strongest(fits, len(expressions))
        residuals = [residual for _, residual, _ in fits.values()]
        room = int(np.count_nonzero(strongest)) if capacity is None else capacity
        order = sorted(
            range(len(expressions)), key=lambda at: (strongest[at] != 0.0, self._gradient(kept[at][0], *residuals))
        )
        dropped = set(order[: max(0, len(expressions) - room)])
        for at in dropped:
            keys.discard(kept[at][0].tobytes())
        expressions[:] = [expression for at, expression in enumerate(expressions) if at not in dropped]
        kept[:] = [columns for at, columns in enumerate(kept) if at not in dropped]
        return len(dropped)

    def _screening(self, rows: int) -> np.ndarray:
        if rows <= MIN_SCREENING_ROWS:
            return np.arange(rows)
        count = max(MIN_SCREENING_ROWS, int(rows * SCREENING_SHARE))
        return np.unique(np.linspace(0, rows - 1, count).round().astype(int))

    def _compute(self, operator: str, first: np.ndarray, second: np.ndarray | None, value: float | None) -> np.ndarray:
        """The operation's values; a row where an operand is blank (NaN) stays blank."""
        blank = np.isnan(first) if second is None else np.isnan(first) | np.isnan(second)
        with np.errstate(all="ignore"):
            if operator == "abs":
                computed = np.abs(first)
            elif second is None:
                computed = ((first >= value) if operator == ">=" else (first <= value)).astype(float)
            else:
                match operator:
                    case "+":
                        computed = first + second
                    case "-":
                        computed = first - second
                    case "*":
                        computed = first * second
                    case "/":
                        computed = first / np.maximum(1.0, second)
                    case "max":
                        computed = np.maximum(first, second)
                    case "min":
                        computed = np.minimum(first, second)
                    case ">=":
                        computed = (first >= second).astype(float)
                    case "==":
                        computed = (first == second).astype(float)
                    case _:
                        raise ValueError(f"Unknown operator {operator!r}")
        return np.where(blank, np.nan, computed)

    def scaling(self, column: np.ndarray) -> tuple[float, float]:
        """The center and the scale the fit reads the column with. Without blanks, its mean and standard deviation. With
        blanks (NaN), 0 and the root mean square of its values: a blank reads as 0 and adds nothing, and a detector whose
        value is always the same when it fires still tells its rows apart. A scale of 0 when nothing varies."""
        blank = np.isnan(column)
        if not blank.any():
            if not len(column):
                return 0.0, 0.0
            scale = float(column.std())
            return float(column.mean()), scale if math.isfinite(scale) else 0.0
        present = column[~blank]
        scale = float(np.sqrt(np.mean(present**2))) if len(present) else 0.0
        return 0.0, scale if math.isfinite(scale) else 0.0

    def standard(self, column: np.ndarray, scaling: tuple[float, float] | None = None) -> np.ndarray:
        """The column as the fit reads it, a blank reading as 0, centered and scaled by the scaling given or the column's
        own; all zeros at a scale of 0."""
        center, scale = self.scaling(column) if scaling is None else scaling
        if scale == 0.0:
            return np.zeros(len(column))
        return (np.nan_to_num(column, nan=0.0) - center) / scale

    def share(self, column: np.ndarray) -> float:
        """The share of rows where the column isn't blank: where what it reads is there."""
        return float(np.mean(~np.isnan(column))) if len(column) else 0.0

    def _price(self, price: float, expression: Expression, column: np.ndarray) -> float:
        """What a weight on the column costs: the price, per clause, per share of rows where it isn't blank."""
        return price * expression.clauses * self.share(column)

    def _gradient(self, column: np.ndarray, *residuals: np.ndarray) -> float:
        """How steeply the loss falls when the standardized column gets a weight, the steepest over the residuals."""
        if len(column) == 0 or not residuals:
            return 0.0
        standard = self.standard(column)
        return max(abs(float(np.mean(standard * residual))) for residual in residuals)

    def _usable(self, column: np.ndarray) -> bool:
        """Every value finite where not blank, and the column, as the fit reads it, not all zeros."""
        present = column[~np.isnan(column)]
        return bool(np.all(np.isfinite(present))) and bool(np.any(self.standard(column) != 0.0))

    def _digest(self, template: str) -> bytes:
        return hashlib.blake2b(template.encode(), digest_size=8).digest()
