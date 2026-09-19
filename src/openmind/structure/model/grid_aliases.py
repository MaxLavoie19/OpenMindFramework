from typing import Protocol

from openmind.structure.model.coordinates import Coordinates


class GridAliases(Protocol):
    """How a game names a grid's cells, such as chess's a1 … h8. Coordinates stay (row, column) from the top; the alias
    decides how its names map onto them. It must be hashable and picklable, since states hold it."""

    def to_coordinates(self, alias: str) -> Coordinates: ...

    def to_alias(self, where: Coordinates) -> str: ...
