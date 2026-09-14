import numpy as np

from openmind.rbs.model.row_arrays import RowArrays


class AdvantageContrast:
    """Compares, state by state, the actions in a scope that a condition matches with the other actions in the scope."""

    def differences(self, arrays: RowArrays, scope: np.ndarray, matching: np.ndarray) -> np.ndarray:
        """For every state where both kinds of actions occur, the visit-weighted mean advantage of the matching ones
        minus that of the others, in state order."""
        size = int(arrays.states.max()) + 1 if len(arrays.states) else 0
        inside_weights, inside_sums = self._weighted(arrays, scope & matching, size)
        outside_weights, outside_sums = self._weighted(arrays, scope & ~matching, size)
        both = (inside_weights > 0) & (outside_weights > 0)
        return inside_sums[both] / inside_weights[both] - outside_sums[both] / outside_weights[both]

    def _weighted(self, arrays: RowArrays, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        states, visits = arrays.states[mask], arrays.visits[mask]
        weights = np.bincount(states, weights=visits, minlength=size)
        sums = np.bincount(states, weights=visits * arrays.advantages[mask], minlength=size)
        return weights, sums
