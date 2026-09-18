from dataclasses import dataclass

from openmind.inference.model.symbol import PredicateSymbol
from openmind.inference.model.term import Term, Variable


@dataclass(frozen=True, slots=True)
class Atom:
    """A predicate applied to its arguments."""

    symbol: PredicateSymbol
    arguments: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class Equal:
    """Two terms are equal."""

    left: Term
    right: Term


@dataclass(frozen=True, slots=True)
class Compare:
    """Two numbers compared by `<=`, `<`, `>=` or `>`."""

    operator: str
    left: Term
    right: Term


@dataclass(frozen=True, slots=True)
class Truth:
    """True or false."""

    value: bool


@dataclass(frozen=True, slots=True)
class Not:
    body: "Formula"


@dataclass(frozen=True, slots=True)
class And:
    parts: tuple["Formula", ...]


@dataclass(frozen=True, slots=True)
class Or:
    parts: tuple["Formula", ...]


@dataclass(frozen=True, slots=True)
class Implies:
    premise: "Formula"
    conclusion: "Formula"


@dataclass(frozen=True, slots=True)
class Iff:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True, slots=True)
class Equisatisfiable:
    """In a proof's own steps: two formulas that are satisfiable together or not at all, as a prover's normal forms are."""

    left: "Formula"
    right: "Formula"


@dataclass(frozen=True, slots=True)
class ForAll:
    variables: tuple[Variable, ...]
    body: "Formula"


@dataclass(frozen=True, slots=True)
class Exists:
    variables: tuple[Variable, ...]
    body: "Formula"


type Formula = Atom | Equal | Compare | Truth | Not | And | Or | Implies | Iff | Equisatisfiable | ForAll | Exists
