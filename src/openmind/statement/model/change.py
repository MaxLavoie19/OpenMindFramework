from dataclasses import dataclass
from typing import ClassVar

from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Placed:
    """Something put where there was nothing, or over what was there."""

    model: str
    at: Coordinates
    value: Value

    #: Which of its places whatever stood there stops standing at, named so that everything needing to know
    #: reads one declaration rather than its own copy of this docstring.
    LOSES: ClassVar[str] = "at"

    @property
    def losing(self) -> Coordinates | None:
        """Where whatever stood there stops standing there, or None where nothing can be lost.

        **Said here because more than one thing has to agree about it, and prose is not a place to keep an
        agreement.** What each kind of change means was written in these docstrings and read separately by
        everything that cared: by what applies a change to a board, by what says it in words, and by what asks
        whether something was taken. The third asked about removals alone — and a predictor that described a
        capture as an arrival, which makes the same board out of every position so nothing preferred either,
        became a predictor in which nothing could ever be taken, silently, and a king who was never in danger.

        Where, and not whether. Whether anything was actually lost is a question about the board, which
        whoever holds the board can answer and this cannot."""
        return getattr(self, self.LOSES) if self.LOSES else None


@dataclass(frozen=True, slots=True)
class Removed:
    """What stood there, gone."""

    model: str
    at: Coordinates

    LOSES: ClassVar[str] = "at"

    @property
    def losing(self) -> Coordinates | None:
        """Where whatever stood there stops standing there. See `Placed.losing`."""
        return getattr(self, self.LOSES) if self.LOSES else None


@dataclass(frozen=True, slots=True)
class Moved:
    """What stood at one place now at another, and nothing where it was."""

    model: str
    source: Coordinates
    target: Coordinates

    #: The target and not the source. What stood at the source is not lost but elsewhere, and a king walking
    #: off a square has not been captured.
    LOSES: ClassVar[str] = "target"

    @property
    def losing(self) -> Coordinates | None:
        """Where whatever stood there stops standing there: the target, which this arrives over."""
        return getattr(self, self.LOSES) if self.LOSES else None


@dataclass(frozen=True, slots=True)
class Told:
    """A scalar of the position now reading otherwise."""

    model: str
    value: Value

    #: Nothing: nothing stands on a scalar, so nothing can stop standing there.
    LOSES: ClassVar[str] = ""

    @property
    def losing(self) -> Coordinates | None:
        """None: nothing stands on a scalar, so nothing can stop standing there."""
        return getattr(self, self.LOSES) if self.LOSES else None


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


#: Every kind of change there is, so anything reading them by the name a consequence calls its kind has one
#: list to read rather than its own.
CHANGES: tuple[type, ...] = (Placed, Removed, Moved, Told)

#: Those of them that can leave a thing no longer standing where it stood, by that name.
#:
#: Derived rather than written out, because it is the same agreement `losing` is and a second copy of an
#: agreement is how the two halves came to disagree in the first place. A kind added later is one this knows.
LOSING: tuple[str, ...] = tuple(one.__name__ for one in CHANGES if one.LOSES)
