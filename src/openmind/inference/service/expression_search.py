import gc
import hashlib
import itertools
import logging
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np
from scipy.special import expit

from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import CANDIDATE_BATCH, MIN_SCREENING_ROWS, SCREENING_SHARE
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


class ExpressionSearch:
    """Searches expressions that value positions, for any domain, from the leaves up, within a budget of time, memory and
    candidates. Each generation:

    1. fits the kept expressions at the price, a price per clause, and takes the residual of the scaled payoffs;
    2. expands every kept expression, the weighted first: its look-aheads, absolute value, thresholds, pattern and
       aggregate children once, and its combinations with every kept expression it hasn't met; then the expressions
       passed over since, the steepest first, into their look-aheads, pattern and aggregate children. Parents take turns,
       one candidate each, and candidates are made only as they are tried, never all at once;
    3. tries the candidates in batches: a candidate is evaluated on the screening rows and admitted only when its
       standardized gradient against the residual is above the price times its clauses, that is when the fit would give
       it a weight; then it is evaluated on every row and kept when it still is, varies, and repeats no kept column.
       Thresholds, absolute values and combinations are computed from the kept columns; the others are evaluated as
       rules, in the evaluator's workers;
    4. evicts, when the kept columns and the fit's copies of them would pass the memory budget, the unweighted
       expressions with the smallest gradients first.

    Memory is measured, not estimated. Before each batch, when this process holds more than the memory budget, the search
    clears the views, forgets the expressions passed over and evicts every unweighted column; if the process still holds
    more, it stops. Every process evaluating terms clears its views once it holds more than its share of the budget.

    It stops when the time, memory or candidate budget runs out, or when a generation has nothing left to try. A
    generation that keeps nothing doesn't stop it: the next one builds on the expressions it passed over."""

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
        targets: np.ndarray,
        price: float,
        max_steps: int,
        tolerance: float,
        budget: SearchBudget,
        seeds: Sequence[Expression] = (),
    ) -> ExpressionSearchResult:
        """Targets are the training rows' payoffs scaled from 0 to 1. Seeds, such as the expressions a deduction induced,
        are tried in the first generation before the leaves, in their order."""
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
            weights, residual, loss = self._fit(expressions, kept, targets, price, max_steps, tolerance)
            candidates: Iterator[Candidate] = (
                iter([(expression, None, None, None, None) for expression in first])
                if generation == 1
                else self._expand(expressions, kept, weights, residual, vocabulary, expanded, combined, passed_over)
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
                    evicted += self._free(expressions, kept, keys, passed_over, used, budget, targets, price, max_steps, tolerance)
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
                        domain, batch, training, held_out, screening, screen_rows, residual, price
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
                    evicted += self._evict(expressions, kept, keys, capacity, targets, price, max_steps, tolerance)
            logger.info(
                "Generation %d: %d candidates tried, %d kept, %d evicted; %d expressions kept, %d weighted at price %s, "
                "looking up to %d actions ahead; training loss %s before the generation; %d candidates tried in all; "
                "%.0f seconds left; %d bytes held",
                generation,
                tried_count,
                admitted,
                evicted,
                len(expressions),
                int(np.count_nonzero(weights)),
                price,
                max((expression.plies for expression in expressions), default=0),
                loss,
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
        targets: np.ndarray,
        price: float,
        max_steps: int,
        tolerance: float,
    ) -> int:
        """Clears the views, forgets the expressions passed over and evicts every unweighted column; the columns
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
        residual: np.ndarray,
        price: float,
    ) -> tuple[list[tuple[Expression, Columns]], list[tuple[Expression, float]]]:
        """The candidates admitted, with their columns, and those that varied on the screening rows without being
        admitted, with their gradient there."""
        screen_residual = residual[screening]
        everything = len(screening) == len(training)
        found: list[tuple[Expression, Columns]] = []
        passed: list[tuple[Expression, float]] = []
        for expression, operator, first, second, value in batch:
            if operator is None or first is None:
                continue
            threshold = price * expression.clauses
            screened = self._compute(operator, first[0][screening], None if second is None else second[0][screening], value)
            gradient = self._gradient(screened, screen_residual)
            if gradient <= threshold:
                if gradient > 0.0:
                    passed.append((expression, gradient))
                continue
            train = self._compute(operator, first[0], None if second is None else second[0], value)
            if self._usable(train) and self._gradient(train, residual) > threshold:
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
            gradient = self._gradient(column, screen_residual)
            if gradient > price * expression.clauses:
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
                if self._gradient(train, residual) > price * expression.clauses:
                    found.append((expression, (train, held)))
                    continue
            passed.append((expression, self._gradient(column, screen_residual)))
        return found, passed

    def _expand(
        self,
        expressions: Sequence[Expression],
        kept: Sequence[Columns],
        weights: np.ndarray,
        residual: np.ndarray,
        vocabulary: Vocabulary,
        expanded: set[bytes],
        combined: dict[bytes, set[bytes]],
        passed_over: list[tuple[Expression, float]],
    ) -> Iterator[Candidate]:
        """Every kept expression's candidates, the weighted first, then those of the expressions passed over since the
        last expansion, the steepest first; the parents take turns, and each makes its candidates only when asked."""
        gradients = [self._gradient(columns[0], residual) for columns in kept]
        order = sorted(range(len(expressions)), key=lambda at: (weights[at] == 0.0, -abs(weights[at]), -gradients[at]))
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
                for child, relation, cut in generator.thresholds(expression, columns[0].tolist())
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
        matrix = np.column_stack([columns[0] for columns in kept])
        standard = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        costs = np.array([expression.clauses for expression in expressions], dtype=float)
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
        targets: np.ndarray,
        price: float,
        max_steps: int,
        tolerance: float,
    ) -> int:
        """Keeps at most capacity expressions, or only the weighted ones without a capacity, dropping the unweighted with
        the smallest gradients first; the number dropped."""
        weights, residual, _ = self._fit(expressions, kept, targets, price, max_steps, tolerance)
        room = int(np.count_nonzero(weights)) if capacity is None else capacity
        order = sorted(range(len(expressions)), key=lambda at: (weights[at] != 0.0, self._gradient(kept[at][0], residual)))
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
        if operator == "abs":
            return np.abs(first)
        if second is None:
            return ((first >= value) if operator == ">=" else (first <= value)).astype(float)
        with np.errstate(all="ignore"):
            match operator:
                case "+":
                    return first + second
                case "-":
                    return first - second
                case "*":
                    return first * second
                case "/":
                    return first / np.maximum(1.0, second)
                case "max":
                    return np.maximum(first, second)
                case "min":
                    return np.minimum(first, second)
                case ">=":
                    return (first >= second).astype(float)
                case "==":
                    return (first == second).astype(float)
        raise ValueError(f"Unknown operator {operator!r}")

    def _gradient(self, column: np.ndarray, residual: np.ndarray) -> float:
        """How steeply the loss falls when the standardized column gets a weight."""
        if len(column) == 0:
            return 0.0
        scale = float(column.std())
        if scale == 0.0 or not np.isfinite(scale):
            return 0.0
        return abs(float(np.mean((column - column.mean()) / scale * residual)))

    def _usable(self, column: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(column))) and float(np.ptp(column)) != 0.0

    def _digest(self, template: str) -> bytes:
        return hashlib.blake2b(template.encode(), digest_size=8).digest()
