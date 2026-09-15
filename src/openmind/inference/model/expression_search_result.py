from dataclasses import dataclass

import numpy as np

from openmind.inference.model.expression import Expression


@dataclass(frozen=True, slots=True)
class ExpressionSearchResult:
    """The expressions a search kept, each with its values on the training rows and on the held-out rows, in the same
    order; how many generations it ran; why it stopped; and how many candidates it tried."""

    expressions: tuple[Expression, ...]
    training: tuple[np.ndarray, ...]
    held_out: tuple[np.ndarray, ...]
    generations: int
    stopped: str
    tried: int
