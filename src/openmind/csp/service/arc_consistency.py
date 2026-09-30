from collections import deque
from collections.abc import Iterable

from openmind.csp.model.constraint_index import ConstraintIndex
from openmind.csp.model.support_table import SupportTable
from openmind.csp.repository.domain_repository import DomainRepository


class ArcConsistency:
    """AC-3 over support tables: removes every value that has no allowed partner left in a table's other variable.

    It narrows the repository it is given and answers nothing — what it changed is on the trail, which is where the
    search reads it from. The index is built once for a search rather than rebuilt here per call."""

    def propagate(self, repository: DomainRepository, index: ConstraintIndex, changed: Iterable[str]) -> None:
        """Narrows the domains starting from the changed variables. Raises Wipeout when a domain empties."""
        queue = deque(
            (table, self._other(table, variable))
            for variable in changed
            for table in index.tables_of(variable)
        )
        while queue:
            table, target = queue.popleft()
            unsupported = self._unsupported(table, target, repository)
            if not unsupported:
                continue
            for value in unsupported:
                repository.remove(target, value)
            queue.extend(
                (other, self._other(other, target)) for other in index.tables_of(target) if other is not table
            )

    def _unsupported(self, table: SupportTable, target: str, repository: DomainRepository) -> list[object]:
        if target == table.first:
            partners = repository.values[table.second]
            return [
                value
                for value in repository.values[target]
                if not any((value, partner) in table.allowed for partner in partners)
            ]
        partners = repository.values[table.first]
        return [
            value
            for value in repository.values[target]
            if not any((partner, value) in table.allowed for partner in partners)
        ]

    def _other(self, table: SupportTable, variable: str) -> str:
        return table.second if variable == table.first else table.first
