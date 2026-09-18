from openmind.world.model.value import Value

#: A cell's coordinates: a whole number on a grid of one dimension, a tuple of whole numbers otherwise.
type Coordinates = int | tuple[int, ...]


class Grid(dict[Coordinates, Value]):
    """The variables of a base whose indices are all whole numbers, by coordinates. Nothing here knows a game. It is a
    dict, so rules read and write it as they do any base."""

    __slots__ = ()

    def __reduce__(self) -> tuple[type["Grid"], tuple[dict[Coordinates, Value]]]:
        """A copy sent to another process is built from its cells."""
        return (Grid, (dict(self),))

    def where(self, value: Value) -> tuple[Coordinates, ...]:
        """The coordinates holding the value, in order."""
        return tuple(sorted(key for key, held in self.items() if held == value))  # type: ignore[type-var]
