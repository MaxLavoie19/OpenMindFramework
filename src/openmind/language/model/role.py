from dataclasses import dataclass

from openmind.language.model.shape import Shape


@dataclass(frozen=True, slots=True)
class Role:
    """One job a notation gives a place, across every shape that has it.

    Held as the places themselves rather than a name, because the name would be ours. A role covering the last
    square of four shapes is one saying; four roles covering one each is a table of four, and the difference
    between those is the difference between a grammar and a list."""

    places: tuple[tuple[Shape, int], ...]

    def filled_by(self, shape: Shape | None, said: str) -> str | None:
        """Which character stands in this role in that notation, or None where the role has no place in it."""
        if shape is None:
            return None
        for one, at in self.places:
            if one == shape:
                return said[at] if at < len(said) else None
        return None

    @property
    def readable(self) -> str:
        shape, at = self.places[0]
        held = f"place {at + 1} of {shape.readable}"
        return held if len(self.places) == 1 else f"{held}, and {len(self.places) - 1} more"
