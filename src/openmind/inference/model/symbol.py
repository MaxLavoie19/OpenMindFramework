from dataclasses import dataclass

from openmind.inference.model.sort import Sort


@dataclass(frozen=True, slots=True)
class FunctionSymbol:
    """A function by name, with the sorts of its arguments and of its result; a constant is a function of no argument."""

    name: str
    arguments: tuple[Sort, ...]
    result: Sort


@dataclass(frozen=True, slots=True)
class PredicateSymbol:
    """A relation by name, with the sorts of its arguments: applied, it is true or false."""

    name: str
    arguments: tuple[Sort, ...]
