from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionParameter:
    """The value of one of the action's parameters."""

    name: str
