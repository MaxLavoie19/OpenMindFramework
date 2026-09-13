from dataclasses import dataclass

from openmind.csp.model.action_definition import ActionDefinition


@dataclass(frozen=True, slots=True)
class Problem:
    """A constraint satisfaction problem: the actions a domain allows."""

    actions: tuple[ActionDefinition, ...]
