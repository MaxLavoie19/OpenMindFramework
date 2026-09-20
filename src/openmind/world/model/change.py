from dataclasses import dataclass

from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Placed:
    """Something put where there was nothing, or over what was there."""

    model: str
    at: Coordinates
    value: Value


@dataclass(frozen=True, slots=True)
class Removed:
    """What stood there, gone."""

    model: str
    at: Coordinates


@dataclass(frozen=True, slots=True)
class Moved:
    """What stood at one place now at another, and nothing where it was."""

    model: str
    source: Coordinates
    target: Coordinates


@dataclass(frozen=True, slots=True)
class Told:
    """A scalar of the position now reading otherwise."""

    model: str
    value: Value


#: One change an action makes to a position.
#:
#: An effect that hands back a position says what the position became and never what the action did. Whatever
#: differs between the two has to be worked out, and which difference was the point cannot be: a move that takes a
#: piece and a move onto an empty square differ in the same two squares, and a pawn taken in passing stands on
#: neither of the squares the move names. Said as changes, an action removes something, or puts something down, or
#: carries it from here to there, and says where — so what it did is there to be read rather than inferred.
#:
#: Changes are made in the order they are given. Removing what stands on a square and then moving onto it takes a
#: piece; moving onto it and then removing what stands there takes the piece that just arrived.
Change = Placed | Removed | Moved | Told
