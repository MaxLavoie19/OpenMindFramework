import math

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import AGGREGATE_INDEX, HERE, MEMORY_CHECK_INTERVAL, VIEW
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner

#: What a reading gives where it can't be read, or gives something other than a finite number.
UNREADABLE = None


class ReadingCache:
    """What a reading gives at every index of a base, for a position and a player, kept so that every candidate reading
    it reads it once.

    A generation's candidates are mostly one another's bodies with one more reading, so the same reading is asked for by
    dozens of candidates on the same rows. A reading is Python source over the index `i`, as an aggregate's body records
    its parts (see `inference/model/aggregate.py`). Its values come back in the base's index order, with None where the
    reading raises or gives something other than a finite number. Like the views, the cache is cleared whenever the
    process holds more than its share of memory, and by the process's memory guard."""

    def __init__(
        self,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
        memory_meter: MemoryMeter,
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._memory_meter = memory_meter
        self._memory_share = math.inf
        self._remembered = 0
        self._values: dict[tuple[object, str, str | None, str], tuple[float | None, ...]] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        """What a reading gives is this process's own: a copy starts with nothing kept."""
        return {name: value for name, value in self.__dict__.items() if name not in ("_values", "_memory_guard")}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._values = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def memory_entries(self) -> int:
        return len(self._values)

    def evict_memory(self, entries: int) -> None:
        evict_oldest(self._values, entries)

    def clear_memory(self) -> None:
        self.clear()

    def limit_memory(self, memory_bytes: int) -> None:
        """The share of memory this process holds before the readings are forgotten."""
        self._memory_share = memory_bytes

    def clear(self) -> None:
        """Forgets every reading kept."""
        self._values.clear()

    def values(self, domain: Domain, row: PositionRow, base: str, reading: str) -> tuple[float | None, ...]:
        """The reading's value at every index of the base, in the base's order, for the row's position read for the row's
        player; kept for that position, player and reading."""
        key = (row.state, row.player, base, reading)
        kept = self._values.get(key)
        if kept is None:
            kept = self._read(domain, row, base, reading)
            self._remember(key, kept)
        return kept

    def _read(self, domain: Domain, row: PositionRow, base: str, reading: str) -> tuple[float | None, ...]:
        source = f"[{reading} for {AGGREGATE_INDEX} in {VIEW}.{base}]".replace(VIEW, HERE)
        compiled = self._rule_compiler.compile_value(PythonRule(source))
        names = self._consequence_library.names(domain, row.state, row.player)
        try:
            given = self._rule_runner.value(compiled, row.state, None, names)
        except (LookupError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
            return ()
        return tuple(self._number(value) for value in given)  # type: ignore[union-attr]

    def _number(self, value: object) -> float | None:
        if isinstance(value, bool | int | float | np.bool_ | np.number):
            number = float(value)  # type: ignore[arg-type]
            return number if math.isfinite(number) else UNREADABLE
        return UNREADABLE

    def _remember(self, key: tuple[object, str, str | None, str], values: tuple[float | None, ...]) -> None:
        self._remembered += 1
        if self._remembered % MEMORY_CHECK_INTERVAL == 0 and self._memory_meter.resident_bytes() > self._memory_share:
            self._values.clear()
        self._memory_guard.remembered()
        self._values[key] = values
