from dataclasses import dataclass

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.support_table import SupportTable
from openmind.csp.model.variable import Variable


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """What backtracking explores for one action: its variables with their remaining domains, and its constraints in the
    forms propagation uses."""

    action: str
    variables: tuple[Variable, ...]
    tables: tuple[SupportTable, ...]
    groups: tuple[AllDifferentGroup, ...]
    constraints: tuple[ScopedConstraint, ...]
