from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PythonRule:
    """A rule written in Python: an expression, or a whole script. Any Python is allowed, imports and libraries included;
    a rule reads the state's variables, the action's parameters by name, and the names its domain's definitions create."""

    source: str
