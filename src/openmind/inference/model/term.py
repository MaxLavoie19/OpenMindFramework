from dataclasses import dataclass

from openmind.inference.model.sort import Sort
from openmind.inference.model.symbol import FunctionSymbol


@dataclass(frozen=True, slots=True)
class Variable:
    """A variable of a sort, bound by a quantifier or free in a clause."""

    name: str
    sort: Sort


@dataclass(frozen=True, slots=True)
class Constant:
    """A named thing of a sort, such as a particular set."""

    name: str
    sort: Sort


@dataclass(frozen=True, slots=True)
class Number:
    """A whole or real number."""

    value: int | float


@dataclass(frozen=True, slots=True)
class Application:
    """A function applied to its arguments; an argument may be a term or, in a proof's own steps, a formula."""

    symbol: FunctionSymbol
    arguments: tuple[object, ...]


type Term = Variable | Constant | Number | Application
