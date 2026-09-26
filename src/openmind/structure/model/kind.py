from dataclasses import dataclass, replace
from itertools import product

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Kind:
    """What a value may be: a closed set of things, or named parts each of a kind, and possibly nothing.

    **A game declares its kinds, and that is where domains come from.** Without this, the only way to know what a
    square may hold is to look at what squares have held so far, and a learner that does that learns the positions
    it has seen. A game that says up front that a piece is one of six types lets OMF say "any type but a rook"
    the first time it sees a rook, rather than after it has met all six.

    It says nothing about what the values mean. `("white", "black")` is two things that are not each other, and
    which of them moves first, or upward, or owns what, is not here and is to be learned.

    A kind with `nothing` may be absent — an empty square holds no piece — which is how a game says that a part is
    optional without inventing a value meaning "none"."""

    name: str
    values: tuple[Value, ...] = ()
    parts: tuple[tuple[str, "Kind"], ...] = ()
    nothing: bool = False
    builds: type | None = None

    def or_nothing(self) -> "Kind":
        """The same kind, which may also be absent."""
        return replace(self, nothing=True)

    def part(self, name: str) -> "Kind":
        for held, kind in self.parts:
            if held == name:
                return kind
        raise KeyError(f"{self.name} has no part called {name!r}")

    @property
    def domain(self) -> tuple[Value, ...]:
        """Every value of this kind, where they can be listed: its own, and nothing where it may be absent.

        A kind of parts can be listed too, where the game said what such a thing is made with: what it may be is
        every combination of what its parts may be. That is how a parameter pointing at a cell gets its
        sixty-four values from two ranges of eight, without anything having to know that a board is square or that
        cells are what is being counted.

        Without a maker there is nothing to build the combinations into, and the domain is empty — a kind of parts
        the game never said how to make is a description, not a thing it can hand out."""
        if not self.parts:
            return (*self.values, *((None,) if self.nothing else ()))
        made = (
            ()
            if self.builds is None
            else tuple(
                self.builds(*combination)
                for combination in product(*(kind.domain for _, kind in self.parts))
            )
        )
        return (*made, *((None,) if self.nothing else ()))
