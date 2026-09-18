import math
from collections.abc import Sequence

import numpy as np

from openmind.inference.constant.inference_constant import COUNT, HIGHEST, LOWEST, SUM
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.reading_cache import ReadingCache

#: What an aggregate is folded from: its base, its kind, whether it reads pairs of indices, its readings and the
#: operations joining them.
type AggregateParts = tuple[str, str, bool, tuple[str, ...], tuple[str, ...]]
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner


class TermEvaluator:
    """Evaluates terms on position rows. On each row a term reads the row's state and the consequence library's names,
    `me` being the row's player. Several terms at once are evaluated in the task runner's workers: the rows are split
    into slices, and each worker evaluates every term on its slice."""

    def __init__(
        self,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
        task_runner: TaskRunner,
        reading_cache: ReadingCache | None = None,
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._task_runner = task_runner
        self._reading_cache = reading_cache

    def limit_memory(self, memory_bytes: int) -> None:
        """Every process evaluating terms, this one or a worker, clears its views and its readings once it holds more
        than an even share of that many bytes between the task runner's workers."""
        share = memory_bytes // self._task_runner.workers
        self._consequence_library.limit_memory(share)
        if self._reading_cache is not None:
            self._reading_cache.limit_memory(share)

    def clear_memory(self) -> None:
        """Forgets what this process's consequence library and readings kept."""
        self._consequence_library.clear_memory()
        if self._reading_cache is not None:
            self._reading_cache.clear()

    def aggregate_columns(
        self, rbs: RuleBasedSystem, rows: Sequence[PositionRow], parts: Sequence[AggregateParts]
    ) -> list[np.ndarray | None]:
        """What each aggregate gives on every row, folded from its readings instead of run as one expression: a reading
        is read once per position and reused by every aggregate that reads it. Parts are `(base, kind, pair, readings,
        operations)`; an aggregate over pairs of indices, or one whose parts weren't recorded, gives None, since a
        reading kept per cell can't answer it. Without a reading cache, every column is None."""
        if self._reading_cache is None:
            return [None] * len(parts)
        return [self._folded(rbs, rows, part) for part in parts]

    def _folded(self, rbs: RuleBasedSystem, rows: Sequence[PositionRow], parts: AggregateParts) -> np.ndarray | None:
        base, kind, pair, readings, operations = parts
        if pair or not readings or len(operations) != len(readings) - 1:
            return None
        cache = self._reading_cache
        column = np.empty(len(rows))
        for index, row in enumerate(rows):
            cells: list[float] | None = None
            for at, reading in enumerate(readings):
                values = cache.values(rbs, row, base, reading)  # type: ignore[union-attr]
                if any(value is None for value in values):
                    return None
                if cells is None:
                    cells = list(values)  # type: ignore[arg-type]
                    continue
                if len(values) != len(cells):
                    return None
                cells = [self._combined(operations[at - 1], held, value) for held, value in zip(cells, values, strict=True)]  # type: ignore[arg-type]
            column[index] = self._aggregated(kind, cells or [])
        return column

    def _combined(self, operation: str, held: float, value: float) -> float:
        """A body grown by an operation and a reading, as `ExpressionGenerator._body` writes it."""
        match operation:
            case "+":
                return held + value
            case "-":
                return held - value
            case "*":
                return held * value
            case "/":
                return held / max(1.0, value)
            case "abs":
                return abs(held - value)
            case ">=":
                return float(held >= value)
            case "<=":
                return float(held <= value)
            case "==":
                return float(held == value)
            case "and":
                return value if held else held
            case "or":
                return held if held else value
        raise ValueError(f"Unknown body operation {operation!r}")

    def _aggregated(self, kind: str, cells: Sequence[float]) -> float:
        """The body's values over the base's indices, as `ExpressionGenerator._aggregate_template` writes them."""
        if kind == COUNT:
            return float(sum(1 for value in cells if value))
        if kind == SUM:
            return math.fsum(cells)
        if kind == LOWEST:
            return float(min(cells, default=0))
        if kind == HIGHEST:
            return float(max(cells, default=0))
        raise ValueError(f"Unknown aggregate kind {kind!r}")

    def column(self, rbs: RuleBasedSystem, rows: Sequence[PositionRow], term: PythonRule) -> np.ndarray | None:
        """The term's value on every row, a boolean counting as 0 or 1, and NaN on a row where the term gives None: what
        it reads isn't there at that moment, as a fork detector without a fork. None when the term raises KeyError,
        NameError, TypeError, AttributeError, ValueError or an arithmetic error on a row, or gives something other than a
        finite number or None."""
        compiled = self._rule_compiler.compile_value(term)
        column = np.empty(len(rows))
        for index, row in enumerate(rows):
            try:
                value = self._rule_runner.value(
                    compiled, row.state, None, self._consequence_library.names(rbs, row.state, row.player)
                )
            except (KeyError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
                return None
            if value is None:
                column[index] = np.nan
                continue
            number = self.number(value)
            if number is None:
                return None
            column[index] = number
        return column

    def columns(
        self, rbs: RuleBasedSystem, rows: Sequence[PositionRow], terms: Sequence[PythonRule]
    ) -> list[np.ndarray | None]:
        """What column gives for each term, in the terms' order, the rows split between the task runner's workers."""
        slices = self._task_runner.split(rows)
        if len(slices) <= 1 or not terms:
            return self.slice_columns(rbs, rows, terms)
        count = len(slices)
        results = self._task_runner.map(self.slice_columns, [rbs] * count, slices, [tuple(terms)] * count)
        merged: list[np.ndarray | None] = []
        for index in range(len(terms)):
            parts = [result[index] for result in results]
            merged.append(None if any(part is None for part in parts) else np.concatenate(parts))  # type: ignore[arg-type]
        return merged

    def slice_columns(
        self, rbs: RuleBasedSystem, rows: Sequence[PositionRow], terms: Sequence[PythonRule]
    ) -> list[np.ndarray | None]:
        """What column gives for each term on these rows, in this process."""
        return [self.column(rbs, rows, term) for term in terms]

    def number(self, value: object) -> float | None:
        """A term's value as a number, a boolean counting as 0 or 1; None for anything else or a number that isn't
        finite."""
        kind = type(value)
        if kind is float:
            return value if math.isfinite(value) else None  # type: ignore[arg-type,return-value]
        if kind is int or kind is bool:
            return float(value)  # type: ignore[arg-type]
        if not isinstance(value, bool | int | float | np.bool_ | np.number):
            return None
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) else None
