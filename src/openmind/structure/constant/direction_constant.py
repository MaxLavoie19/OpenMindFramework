from itertools import product

from openmind.structure.model.coordinates import Coordinates


def orthogonal(dimensions: int = 2) -> tuple[Coordinates, ...]:
    """The steps along one axis at a time: up, down, left and right in two dimensions."""
    steps: list[Coordinates] = []
    for axis in range(dimensions):
        for sign in (-1, 1):
            steps.append(tuple(sign if index == axis else 0 for index in range(dimensions)))
    return tuple(steps)


def diagonal(dimensions: int = 2) -> tuple[Coordinates, ...]:
    """The steps along more than one axis at once: the four diagonals in two dimensions."""
    return tuple(
        step for step in product((-1, 0, 1), repeat=dimensions) if sum(1 for part in step if part != 0) > 1
    )


#: OMF's direction sets in two dimensions.
ORTHOGONAL = orthogonal()
DIAGONAL = diagonal()

#: The distances a grid measures.
CHEBYSHEV = "chebyshev"
MANHATTAN = "manhattan"
