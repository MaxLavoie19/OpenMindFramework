from dataclasses import dataclass

from openmind.inference.model.formula import Formula
from openmind.inference.model.symbol import FunctionSymbol, PredicateSymbol


@dataclass(frozen=True, slots=True)
class Axiom:
    """A formula taken as true, by name."""

    name: str
    formula: Formula


@dataclass(frozen=True, slots=True)
class Theory:
    """Definitions by name: the symbols they introduce and the axioms defining them."""

    name: str
    symbols: tuple[FunctionSymbol | PredicateSymbol, ...]
    axioms: tuple[Axiom, ...]
