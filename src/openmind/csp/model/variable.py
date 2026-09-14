from dataclasses import dataclass

from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.state_domain import StateDomain


@dataclass(frozen=True, slots=True)
class Variable:
    """An action parameter to solve for, over fixed values or values computed from the state."""

    name: str
    domain: DiscreteDomain | StateDomain
