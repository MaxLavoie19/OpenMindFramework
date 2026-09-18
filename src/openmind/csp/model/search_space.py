from dataclasses import dataclass

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.support_table import SupportTable
from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """What backtracking explores for one action: each parameter with the values it has left, in the order they were
    given, and the action's constraints in the forms propagation uses."""

    action: str
    variables: tuple[tuple[str, tuple[Value, ...]], ...]
    tables: tuple[SupportTable, ...]
    groups: tuple[AllDifferentGroup, ...]
    constraints: tuple[ScopedConstraint, ...]
