import gc
import hashlib
import itertools
import logging
import math
from types import MappingProxyType
import time
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy.special import expit

from openmind.inference.constant.inference_constant import (
    CANDIDATE_BATCH,
    MIN_SCREENING_ROWS,
    SCREENING_SHARE,
    SEEDED_WEIGHT,
    SINGLE_TARGET,
)
from openmind.inference.model.expression import Expression
from openmind.inference.model.expression_search_result import ExpressionSearchResult
from openmind.inference.model.search_budget import SearchBudget
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.parallel.model.call_over_memory import CallOverMemory
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.sparse_fit import SparseFit
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.sparse_fitter import SparseFitter
from openmind.rbs.service.term_evaluator import AggregateParts, TermEvaluator

logger = logging.getLogger(__name__)

#: A column on the training rows and on the held-out rows.
type Columns = tuple[np.ndarray, np.ndarray]
#: A candidate: its expression; for one computed from kept columns, the operator, the columns it reads and the value of a
#: threshold; None for those of one evaluated as a rule.
type Candidate = tuple[Expression, str | None, Columns | None, Columns | None, float | None]
#: For every target by name: the kept expressions' weights at the price, the residual, and the training loss.
type Fits = dict[str, tuple[np.ndarray, np.ndarray, float]]

#: The kinds a candidate comes in, in the order the log reports them. A candidate's kind says what it cost to
#: try rather than what it says: a `derived` one is arithmetic over columns already computed, where a
#: `look ahead` reads the position after every legal action — in chess upwards of thirty-five evaluations for
#: one candidate. A generation's budget is spent in these proportions, and nothing until now reported them.
LEAF, DERIVED, PATTERN, AGGREGATE, LOOK_AHEAD = "leaf", "derived", "pattern", "aggregate", "look ahead"
CANDIDATE_KINDS = (LEAF, DERIVED, PATTERN, AGGREGATE, LOOK_AHEAD)


class ExpressionSearch:
    """Searches expressions that value positions, for any rbs, from the leaves up, within a budget of time, memory and
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

    Given several targets, every generation fits each target, and a candidate is
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
        rbs: RuleBasedGame,
        training: Sequence[PositionRow],
        held_out: Sequence[PositionRow],
        targets: np.ndarray | Mapping[str, np.ndarray],
        price: float,
        max_steps: int,
        tolerance: float,
        budget: SearchBudget,
        seeds: Sequence[Expression | tuple[Expression, float]] = (),
        dropped: Collection[str] = (),
    ) -> ExpressionSearchResult:
        """Targets are the training rows' payoffs scaled from 0 to 1, or several such targets by name. Seeds, expressions
        given to start from, are tried in the first generation before the leaves, in their order. No target
        raises ValueError.

        `dropped` are templates the search must not take up: a caller that has measured something about a term
        which makes it not worth a generation says so here, and nothing is spent on it or on anything grown
        from it. The search measures nothing of the kind itself — what to drop is the caller's finding, and
        this only honours it. Dropping nothing, which is the default, is the search as it was.

        **A seed may carry the weight the rules imply, and then the weight is half of what it gives.** A linear
        heuristic values a position as the sum of its weighted readings, so the weight on "how many knights I
        have" *is* what a knight is worth — a reasoned number is not carried alongside the term, it starts the
        term off. Given as a bare expression, a seed starts where everything else does.

        **What that weight buys is expansion order, not survival.** A term whose gradient does not pay is
        shrunk to nothing in one step whatever it started at, which is the price doing its work and must stay
        so. What a weighted seed gets is to be *expanded first* — `_expand` sorts by weight, so its conditions,
        its thresholds and its children are generated before anything else's, and the term really wanted is
        reached in the generation that would otherwise be spent working back to it.

        **The weight arrives a generation later than it reads as.** The first generation fits an empty set of
        columns — nothing has been tried yet — so the first fit that can start from a seed's weight is the
        second, once the seeds' own columns are in."""
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
        vocabulary = generator.vocabulary(rbs, (row.state for row in training))
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
        leaves = tuple(one for one in generator.leaves(vocabulary) if one.template not in dropped)
        sown = [one if isinstance(one, tuple) else (one, 0.0) for one in seeds]
        # What the rules said each seeded term is worth, by the template that carries it — so a term dropped,
        # evicted or never admitted simply never asks for it, and nothing has to be kept in step by hand.
        weighed = {expression.template: weight for expression, weight in sown if weight}
        first = tuple(
            {expression.template: expression for expression in (*(one for one, _ in sown), *leaves)}.values()
        )
        logger.info(
            "Searching expressions for %s seconds within %d bytes, trying %s candidates: %d seeds, %d leaves "
            "(%d dropped), %d training rows, %d screened",
            budget.seconds,
            budget.memory_bytes,
            "any number of" if budget.candidates is None else f"at most {budget.candidates}",
            len(seeds),
            len(leaves),
            len(dropped),
            len(training),
            len(screen_rows),
        )
        generation, stopped, tried_total = 0, "", 0
        while not stopped:
            generation += 1
            fits = self._fit_all(expressions, kept, named, price, max_steps, tolerance, weighed)
            residuals = {name: residual for name, (_, residual, _) in fits.items()}
            spending: dict[str, int] = {}
            candidates: Iterator[Candidate] = (
                iter([(expression, None, None, None, None) for expression in first])
                if generation == 1
                else self._expand(
                    expressions, kept, fits, vocabulary, expanded, combined, passed_over, spending
                )
            )
            fresh = (candidate for candidate in candidates if self._digest(candidate[0].template) not in tried)
            tried_count = admitted = evicted = 0
            # Where the generation's candidates went, by what each one cost to try rather than by what it says.
            by_kind: dict[str, list[int]] = {kind: [0, 0, 0, 0] for kind in CANDIDATE_KINDS}
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
                kinds = {}
                for candidate in batch:
                    kind = self._kind(candidate, generation == 1)
                    kinds[candidate[0].template] = kind
                    tally = by_kind[kind]
                    tally[0] += 1
                    if self._crossing(candidate[0]):
                        tally[2] += 1
                try:
                    admissions, passed = self._admitted(
                        rbs, batch, training, held_out, screening, screen_rows, residuals, price
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
                    tally = by_kind.get(kinds.get(expression.template, DERIVED))
                    if tally is not None:
                        tally[1] += 1
                        if self._crossing(expression):
                            tally[3] += 1
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
            self._report_spending(generation, by_kind, spending)
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
        rbs: RuleBasedGame,
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
        screened_columns = self._columns(rbs, screen_rows, rules)
        passing: list[tuple[Expression, np.ndarray]] = []
        for expression, column in zip(rules, screened_columns, strict=True):
            if column is None:
                continue
            gradient = self._gradient(column, *screen_residual)
            if gradient > self._price(price, expression, column):
                passing.append((expression, column))
            elif gradient > 0.0:
                passed.append((expression, gradient))
        if not passing:
            return found, passed
        candidates = [expression for expression, _ in passing]
        trains = [column for _, column in passing] if everything else self._columns(rbs, training, candidates)
        helds = self._columns(rbs, held_out, candidates)
        for (expression, column), train, held in zip(passing, trains, helds, strict=True):
            if train is not None and held is not None and self._usable(train):
                if self._gradient(train, *residual) > self._price(price, expression, train):
                    found.append((expression, (train, held)))
                    continue
            passed.append((expression, self._gradient(column, *screen_residual)))
        return found, passed

    def _columns(
        self, rbs: RuleBasedGame, rows: Sequence[PositionRow], expressions: Sequence[Expression]
    ) -> list[np.ndarray | None]:
        """Each expression's values on the rows: an aggregate whose body recorded its readings is folded from them, every
        reading read once per position however many candidates share it; anything else is run from its source."""
        folded = self._term_evaluator.aggregate_columns(rbs, rows, [self._parts(expression) for expression in expressions])
        left = [expression for expression, column in zip(expressions, folded, strict=True) if column is None]
        if not left:
            return folded
        sources = [self._expression_generator.source(expression) for expression in left]
        ran = iter(self._term_evaluator.columns(rbs, rows, sources))
        return [column if column is not None else next(ran) for column in folded]

    def _parts(self, expression: Expression) -> AggregateParts:
        """What an aggregate is folded from; parts no fold can take for anything else."""
        aggregate = expression.aggregate
        if aggregate is None:
            return ("", "", True, (), ())
        return (aggregate.base, aggregate.kind, aggregate.pair, aggregate.readings, aggregate.operations)

    def _expand(
        self,
        expressions: Sequence[Expression],
        kept: Sequence[Columns],
        fits: Fits,
        vocabulary: Vocabulary,
        expanded: set[bytes],
        combined: dict[bytes, set[bytes]],
        passed_over: list[tuple[Expression, float]],
        spending: dict[str, int] | None = None,
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
        if spending is not None:
            spending["parents"] = len(groups)
            spending["kept parents"] = len(partners)
        passed_over.clear()
        return self._turns(groups, spending)

    def _kept_candidates(
        self,
        expression: Expression,
        columns: Columns,
        partners: Sequence[tuple[Expression, Columns]],
        vocabulary: Vocabulary,
        expanded: set[bytes],
        combined: dict[bytes, set[bytes]],
    ) -> Iterator[Candidate]:
        digest = self._digest(expression.template)
        # **One stream per kind of child, drawn from in turn, because a fixed order lets one kind take
        # everything.** Yielded one kind after another, a parent handed over all fourteen of its look-aheads
        # before its first pattern condition — and with hundreds of parents taking turns, that is thousands of
        # the dearest candidates there are before the cheapest. Measured on a search whose target was white's
        # rooks less black's: of 5,800 candidates tried in a generation, 5,800 were look-aheads, and no term
        # pairing two models was ever reached although only such a term could say what the target was.
        #
        # Taking one from each kind in turn needs no number for what a kind costs. Each kind gets a share of
        # the generation by having a turn, so a parent's pattern conditions are reached in the first rotation
        # whatever else it has to offer. The cheapest go first within a rotation, which is the ordering, and no
        # candidate is removed: a look-ahead is still reached where there is budget for it.
        streams: list[Iterator[Candidate]] = []
        if digest not in expanded:
            expanded.add(digest)
            streams.extend(
                (
                    self._reusing_candidates(expression, columns),
                    self._pattern_candidates(expression, vocabulary),
                    self._aggregate_candidates(expression, vocabulary),
                    self._look_ahead_candidates(expression),
                )
            )
        streams.append(self._combining_candidates(expression, columns, partners, digest, combined))
        yield from self._turns(streams)

    def _reusing_candidates(self, expression: Expression, columns: Columns) -> Iterator[Candidate]:
        """The children that need no new reading: arithmetic over a column already computed."""
        generator = self._expression_generator
        yield from ((child, operator, columns, None, None) for child, operator in generator.unary(expression))
        yield from (
            (child, relation, columns, None, cut)
            for child, relation, cut in generator.thresholds(expression, columns[0][~np.isnan(columns[0])].tolist())
        )

    def _pattern_candidates(self, expression: Expression, vocabulary: Vocabulary) -> Iterator[Candidate]:
        """A condition added to a pattern: one reading of the position as it stands."""
        yield from (
            (child, None, None, None, None)
            for child in self._expression_generator.pattern_children(expression, vocabulary)
        )

    def _aggregate_candidates(self, expression: Expression, vocabulary: Vocabulary) -> Iterator[Candidate]:
        yield from (
            (child, None, None, None, None)
            for child in self._expression_generator.aggregate_children(expression, vocabulary)
        )

    def _look_ahead_candidates(self, expression: Expression) -> Iterator[Candidate]:
        """The dearest children there are: each reads the position again after every legal action."""
        yield from ((child, None, None, None, None) for child in self._expression_generator.look_aheads(expression))

    def _combining_candidates(
        self,
        expression: Expression,
        columns: Columns,
        partners: Sequence[tuple[Expression, Columns]],
        digest: bytes,
        combined: dict[bytes, set[bytes]],
    ) -> Iterator[Candidate]:
        """This expression put together with each other kept expression, a partner at a time.

        Cheap to try, one per partner and so the most numerous kind by far, which is why it takes its turn
        rather than being yielded in a block."""
        generator = self._expression_generator
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
        """An expression nothing weighted, expanded once anyway — by turns over its kinds, as a kept one is.

        This is the path that matters most for a term whose parent shows nothing on its own: counting a kind of
        piece without saying whose it is counts both players' and is nearly constant, so the parent is passed
        over, and only its conditions can say the thing worth saying."""
        yield from self._turns(
            [
                self._pattern_candidates(expression, vocabulary),
                self._aggregate_candidates(expression, vocabulary),
                self._look_ahead_candidates(expression),
            ]
        )

    def _turns(
        self, groups: Iterable[Iterator[Candidate]], spending: dict[str, int] | None = None
    ) -> Iterator[Candidate]:
        """One candidate from each group in turn, until every group is done.

        `spending` is filled as it goes with how many parents were queued and how many were reached: a
        generation that runs out of time partway round leaves the parents after that point unexpanded, and
        whether that is happening is not otherwise visible."""
        live = deque(enumerate(groups))
        reached: set[int] = set()
        while live:
            at, group = live.popleft()
            candidate = next(group, None)
            if candidate is not None:
                live.append((at, group))
                if spending is not None:
                    reached.add(at)
                    spending["reached"] = len(reached)
                yield candidate

    def _fit_all(
        self,
        expressions: Sequence[Expression],
        kept: Sequence[Columns],
        targets: Mapping[str, np.ndarray],
        price: float,
        max_steps: int,
        tolerance: float,
        weighed: Mapping[str, float] = MappingProxyType({}),
    ) -> Fits:
        return {
            name: self._fit(expressions, kept, values, price, max_steps, tolerance, weighed)
            for name, values in targets.items()
        }

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
        weighed: Mapping[str, float] = MappingProxyType({}),
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """The weights at the price, the residual of the scaled payoffs (prediction minus payoff), and the training
        loss.

        `weighed` is what the rules said a seeded term is worth, which the fit starts that term from instead of
        from nothing. The number has to be put on the scale the fit works on: the columns are centered and
        scaled, and the prediction goes through a logistic, so a raw count of fourteen squares means nothing
        here and is multiplied by the column's own scale and held to the range a weight on that scale can
        sensibly take. It is a place to start one fit from, and the fitter moves it or shrinks it away."""
        if not kept:
            mean = float(np.mean(targets))
            prediction = np.full(len(targets), mean)
            clipped = min(max(mean, 1e-12), 1.0 - 1e-12)
            loss = float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)))
            return np.zeros(0), prediction - targets, loss
        standard = np.column_stack([self.standard(columns[0]) for columns in kept])
        costs = np.array(
            [self.cost(expression, columns[0]) for expression, columns in zip(expressions, kept, strict=True)],
            dtype=float,
        )
        fit = self._sparse_fitter.fit(
            standard, targets, price, max_steps, tolerance, self._started(expressions, kept, weighed), costs
        )
        weights = np.asarray(fit.weights, dtype=float)
        prediction = expit(standard @ weights + fit.bias)
        return weights, prediction - targets, self._sparse_fitter.loss(standard, targets, fit.weights, fit.bias)

    def _started(
        self, expressions: Sequence[Expression], kept: Sequence[Columns], weighed: Mapping[str, float]
    ) -> SparseFit | None:
        """Where the fit starts, or nothing where no kept term was seeded with a weight.

        Aligned to the kept columns and never to the order the seeds arrived in, since a term may have been
        evicted, may never have been admitted, or may be one of the leaves. A column that does not vary has a
        scale of nought and is read as all zeros, so there is nothing for a weight to start on there."""
        if not weighed:
            return None
        start = [0.0] * len(expressions)
        for at, (expression, columns) in enumerate(zip(expressions, kept, strict=True)):
            held = weighed.get(expression.template)
            if not held:
                continue
            scale = self.scaling(columns[0])[1]
            if scale:
                start[at] = float(np.clip(held * scale, -SEEDED_WEIGHT, SEEDED_WEIGHT))
        return SparseFit(tuple(start), 0.0, 0, False) if any(start) else None

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

    def firing(self, column: np.ndarray) -> float:
        """The share of rows where the column says something: present, and not nought.

        **Nought is not blank, and charging a term as though it were was dropping detectors.** `share` counts
        the rows where a term isn't blank — where it was read at all — and a mate detector reads **0** on
        almost every position, which counts as speaking. Charged on that, it pays as though it spoke
        everywhere while paying off on the few rows it actually fired.

        Measured, holding the decisive strength at 0.5 and varying only how often the term fires: at price
        0.01 a term firing on 1% of rows came out at **0.000** — dropped outright — while a term firing on 20%
        kept 0.436. Rarity alone decided it. Priced on firing instead, the same term kept 0.489."""
        if not len(column):
            return 0.0
        return float(np.mean(~np.isnan(column) & (column != 0.0)))

    def cost(self, expression: Expression, column: np.ndarray) -> float:
        """What a weight on that term costs: per clause, per share of rows where it isn't blank, per share of
        rows where it fires, per what it takes to read against what the cheapest term here takes.

        **A term is not only worth what it explains, it costs what it takes to read.** A look-ahead reads the
        position after every legal action, so in chess it costs upwards of thirty-five ordinary readings while
        being priced as one more clause. Fitted without that, a heuristic came out taking four tenths of a
        second a position, and a two-second search budget bought four nodes of it — a heuristic that cannot be
        read inside a search is not a heuristic, however well it predicts.

        **Measured and not assumed.** Nothing here knows what a look-ahead is or that chess has thirty-five
        moves; it knows that this term took a hundred times longer than that one on the same rows, which is
        true of whatever made it slow. A term nobody timed costs what its clauses say, as before.

        **Firing multiplies the share rather than replacing it, which is the whole care taken here.** This is
        the adaptive lasso — Zou, JASA 101:1418, whose answer to the lasso's selection inconsistency is to
        weight each coefficient's penalty rather than charge them all alike — and the plain form would charge
        a rare term on its firing alone. That would quietly forgive the reading: a look-ahead that fires once
        in a hundred positions still reads the position after every legal action on all hundred, and the cost
        of reading is real whether the term found anything or not. So both are charged: what it costs to read,
        and how much of what it read it had anything to say about."""
        dearness = self._term_evaluator.dearness(self._expression_generator.source(expression))
        return expression.clauses * self.share(column) * self.firing(column) * dearness

    def _price(self, price: float, expression: Expression, column: np.ndarray) -> float:
        """What a weight on the column costs at that price."""
        return price * self.cost(expression, column)

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

    def _report_spending(
        self, generation: int, by_kind: Mapping[str, Sequence[int]], spending: Mapping[str, int]
    ) -> None:
        """Where a generation's candidates went, and how far round its parents it got.

        **Written to answer one question: whether a cheap candidate that would pair two models is ever
        reached.** A parent makes its children in a fixed order and the parents take turns, so a generation
        that runs out of time partway round leaves the rest unexpanded — and a kind that costs thirty-five
        readings where another costs one can take the whole budget without that being visible anywhere."""
        if not any(tally[0] for tally in by_kind.values()):
            return
        logger.info("Generation %d, what its candidates were and what they cost to try:", generation)
        logger.info("  %-10s %8s %8s %10s %10s", "kind", "tried", "kept", "crossing", "crossing kept")
        for kind in CANDIDATE_KINDS:
            tally = by_kind[kind]
            if tally[0]:
                logger.info("  %-10s %8d %8d %10d %10d", kind, tally[0], tally[1], tally[2], tally[3])
        if spending:
            logger.info(
                "  of %d parents queued to expand (%d of them kept expressions), %d were reached",
                spending.get("parents", 0), spending.get("kept parents", 0), spending.get("reached", 0),
            )

    def _kind(self, candidate: Candidate, seeding: bool) -> str:
        """What a candidate cost to try, which is what its kind says.

        One carrying an operator is arithmetic over columns already computed. The rest are evaluated as rules
        against every row, and of those a pattern or an aggregate reads the position once where a look-ahead
        reads it again after every legal action.

        `seeding` is the first generation, whose candidates are the leaves and the seeds rather than anything
        grown: a leaf reading one cell carries no operator and no shape, and calling it a look-ahead would put
        the cheapest candidates there are in the column for the dearest."""
        expression, operator = candidate[0], candidate[1]
        if operator is not None:
            return DERIVED
        if expression.pattern is not None:
            return PATTERN
        if expression.aggregate is not None:
            return AGGREGATE
        return LEAF if seeding else LOOK_AHEAD

    def _crossing(self, expression: Expression) -> bool:
        """Whether a pattern speaks of more than one model at once — "a rook, and mine" rather than "a rook".

        **The measurement this was written for.** Where a game declares a cell as two grids, as chess declares
        a square as the piece standing there and whose it is, a condition on one of them alone cannot say whose
        anything is: counting rooks counts both players'. Only a pattern pairing two bases says it, and nothing
        reported whether any such candidate was ever reached."""
        pattern = expression.pattern
        if pattern is None:
            return False
        return len({condition.base for condition in pattern.conditions} | {pattern.anchor}) > 1

    def _digest(self, template: str) -> bytes:
        return hashlib.blake2b(template.encode(), digest_size=8).digest()
