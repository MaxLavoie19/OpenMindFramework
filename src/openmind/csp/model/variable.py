from dataclasses import dataclass

from openmind.csp.model.discrete_domain import DiscreteDomain


@dataclass(frozen=True, slots=True)
class Variable:
    """An action parameter to solve for."""

    name: str
    domain: DiscreteDomain
