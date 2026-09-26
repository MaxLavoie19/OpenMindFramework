from dataclasses import dataclass

from openmind.language.model.shape import Shape
from openmind.language.model.sort import Sort


@dataclass(frozen=True, slots=True)
class Grammar:
    """A game's notation as sorts of character and the shapes they fall into.

    What a notation is made of, before anything is known about what it means. Held apart from the couplings on
    purpose: the shapes are a fact about the strings, and a saying measured against a slot of a shape can be
    checked by them rather than agreeing with them by construction."""

    sorts: tuple[Sort, ...]
    shapes: tuple[Shape, ...]

    def sort_of(self, character: str) -> Sort | None:
        """Which sort that character belongs to, or None where it has never been seen."""
        for one in self.sorts:
            if character in one:
                return one
        return None

    def shaped(self, said: str) -> Shape | None:
        """Which shape that notation is, or None where it is one we have not met.

        None is news rather than failure: it says the shapes found so far do not cover what the game just
        played."""
        found = []
        for character in said:
            one = self.sort_of(character)
            if one is None:
                return None
            found.append(one.name)
        shape = Shape(tuple(found))
        return shape if shape in self.shapes else None
