from dataclasses import dataclass

from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Variable


@dataclass(frozen=True, slots=True)
class Atom:
    """One thing said of some terms, before anything has been done to it."""

    literal: Literal


@dataclass(frozen=True, slots=True)
class Truth:
    """True or false outright."""

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
class ForAll:
    variables: tuple[Variable, ...]
    body: "Formula"


@dataclass(frozen=True, slots=True)
class Exists:
    variables: tuple[Variable, ...]
    body: "Formula"


#: How anything is *said* before it is reasoned with.
#:
#: Clauses are what the engine works in, and nobody thinks in clauses. A rule is stated the way it is meant — if
#: this and this then that, for every thing, there is some thing such that — and turning it into clauses is a
#: mechanical job done once, by the clausifier. Keeping the two apart means neither has to bend: whoever writes a
#: rule, or decodes one out of a sentence, never has to know what a normal form is.
type Formula = Atom | Truth | Not | And | Or | Implies | Iff | ForAll | Exists
