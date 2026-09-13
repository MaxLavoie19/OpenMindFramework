from collections import deque
from collections.abc import Iterable

from openmind.csp.model.support_table import SupportTable
from openmind.csp.model.wipeout import Wipeout
from openmind.world.model.value import Value


class ArcConsistency:
    """AC-3 over support tables: removes every value that has no allowed partner left in a table's other variable."""

    def propagate(
        self,
        domains: dict[str, frozenset[Value]],
        tables: tuple[SupportTable, ...],
        changed: Iterable[str],
    ) -> dict[str, frozenset[Value]]:
        """Narrows the domains starting from the changed variables; unchanged domains are kept as they are. Raises Wipeout
        when a domain empties."""
        by_variable: dict[str, list[SupportTable]] = {}
        for table in tables:
            by_variable.setdefault(table.first, []).append(table)
            by_variable.setdefault(table.second, []).append(table)
        narrowed = dict(domains)
        queue = deque(
            (table, self._other(table, variable))
            for variable in changed
            for table in by_variable.get(variable, ())
        )
        while queue:
            table, target = queue.popleft()
            supported = self._supported(table, target, narrowed)
            if len(supported) == len(narrowed[target]):
                continue
            if not supported:
                raise Wipeout(target)
            narrowed[target] = supported
            queue.extend(
                (other, self._other(other, target)) for other in by_variable[target] if other is not table
            )
        return narrowed

    def _supported(
        self, table: SupportTable, target: str, domains: dict[str, frozenset[Value]]
    ) -> frozenset[Value]:
        if target == table.first:
            partners = domains[table.second]
            return frozenset(
                value
                for value in domains[target]
                if any((value, partner) in table.allowed for partner in partners)
            )
        partners = domains[table.first]
        return frozenset(
            value for value in domains[target] if any((partner, value) in table.allowed for partner in partners)
        )

    def _other(self, table: SupportTable, variable: str) -> str:
        return table.second if variable == table.first else table.first
