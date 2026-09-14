from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class RowArrays:
    """Action rows as arrays, one entry per row: the index of its state, shared by the rows of the same state, its
    visits, its advantage and its mean payoff."""

    states: np.ndarray
    visits: np.ndarray
    advantages: np.ndarray
    payoffs: np.ndarray
