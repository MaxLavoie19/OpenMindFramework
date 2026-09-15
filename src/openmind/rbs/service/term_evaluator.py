import math
from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner


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
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._task_runner = task_runner

    def limit_memory(self, memory_bytes: int) -> None:
        """Every process evaluating terms, this one or a worker, clears its views once it holds more than an even share
        of that many bytes between the task runner's workers."""
        self._consequence_library.limit_memory(memory_bytes // self._task_runner.workers)

    def clear_memory(self) -> None:
        """Forgets what this process's consequence library kept."""
        self._consequence_library.clear_memory()

    def column(self, domain: Domain, rows: Sequence[PositionRow], term: PythonRule) -> np.ndarray | None:
        """The term's value on every row, a boolean counting as 0 or 1, and NaN on a row where the term gives None: what
        it reads isn't there at that moment, as a fork detector without a fork. None when the term raises KeyError,
        NameError, TypeError, AttributeError, ValueError or an arithmetic error on a row, or gives something other than a
        finite number or None."""
        compiled = self._rule_compiler.compile_value(term)
        column = np.empty(len(rows))
        for index, row in enumerate(rows):
            try:
                value = self._rule_runner.value(
                    compiled, row.state, None, self._consequence_library.names(domain, row.state, row.player)
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
        self, domain: Domain, rows: Sequence[PositionRow], terms: Sequence[PythonRule]
    ) -> list[np.ndarray | None]:
        """What column gives for each term, in the terms' order, the rows split between the task runner's workers."""
        slices = self._task_runner.split(rows)
        if len(slices) <= 1 or not terms:
            return self.slice_columns(domain, rows, terms)
        count = len(slices)
        results = self._task_runner.map(self.slice_columns, [domain] * count, slices, [tuple(terms)] * count)
        merged: list[np.ndarray | None] = []
        for index in range(len(terms)):
            parts = [result[index] for result in results]
            merged.append(None if any(part is None for part in parts) else np.concatenate(parts))  # type: ignore[arg-type]
        return merged

    def slice_columns(
        self, domain: Domain, rows: Sequence[PositionRow], terms: Sequence[PythonRule]
    ) -> list[np.ndarray | None]:
        """What column gives for each term on these rows, in this process."""
        return [self.column(domain, rows, term) for term in terms]

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
