from dataclasses import dataclass, field

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Variable:
    """A place in a rule that stands for anything.

    This is what a rule made of readings cannot have, and what everything else here is for. A condition comparing
    the name of a reading to a value can only ever say something about the one thing that reading was read of; a
    variable says it of all of them at once. A rule about what a player may do with their own things is then said
    once, rather than once for each player and never as the thing it is about.

    `sort` is what kind of thing it stands for, where the game says. It is not insisted on: a game that says
    nothing leaves it empty, and the readings keep a player and a square apart by being different predicates."""

    name: str
    sort: str = ""


@dataclass(frozen=True, slots=True)
class Constant:
    """A particular thing: a player, a piece kind, a square, whatever a game's values are."""

    name: Value


@dataclass(frozen=True, slots=True)
class Number:
    """A number read as itself, never as one of the values a reading takes."""

    value: int | float


@dataclass(frozen=True, slots=True)
class Functor:
    """A function of terms standing for a term: the square one step on from another, the player whose turn follows.

    A game needs none of these to be reasoned about, and a game that has them can say in one term what would
    otherwise take a variable and a literal to relate it."""

    name: str
    arguments: tuple["Term", ...]

    #: Worked out once when the term is made, and never part of what makes two terms the same.
    #:
    #: A term of terms hashes by hashing everything inside it, and these are looked up constantly — every reading
    #: asked of a case, every pair of terms a generalisation considers. On a game whose squares hold a thing that
    #: holds a thing, the same tree was being walked tens of millions of times over a single position. A term
    #: cannot change, so neither can its hash.
    _hash: int = field(init=False, compare=False, repr=False, default=0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.name, self.arguments)))


def _hashed(held: Functor) -> int:
    return held._hash


# The dataclass writes a hash of its own for a frozen class, so the cached one is put in afterwards rather than
# in the body, where it would be overwritten.
Functor.__hash__ = _hashed  # type: ignore[assignment,method-assign]


#: What something is said of: a variable, a particular thing, a number, or a function of those.
type Term = Variable | Constant | Number | Functor
