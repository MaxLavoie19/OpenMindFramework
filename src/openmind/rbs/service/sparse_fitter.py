import numpy as np
from scipy.special import expit

from openmind.rbs.model.sparse_fit import SparseFit


class SparseFitter:
    """Fits a logistic model with a price on its weights: minimises the mean logistic loss of targets between 0 and 1
    plus price × the sum of the weights' absolute values, the bias unpriced. Accelerated proximal gradient (FISTA): a
    gradient step on the loss, then every weight shrinks toward 0 by step × price and lands on 0 when it would cross it,
    so terms that don't pay their price drop out. The step is 1 / L, L = ‖columns with a bias column‖² / (4 × rows),
    the largest curvature the logistic loss can have, so no step overshoots."""

    def fit(
        self,
        columns: np.ndarray,
        targets: np.ndarray,
        price: float,
        max_steps: int,
        tolerance: float,
        start: SparseFit | None = None,
        costs: np.ndarray | None = None,
    ) -> SparseFit:
        """Starts from start's weights and bias when given, from 0 otherwise; settles once no weight or the bias moves by
        more than tolerance × the largest of 1 and the largest weight. Costs, one per column, multiply the price of that
        column's weight; every weight costs 1 without them."""
        rows, width = columns.shape
        if rows == 0:
            raise ValueError("A fit needs at least one row")
        design = np.hstack([columns, np.ones((rows, 1))])
        step = 4.0 * rows / float(np.linalg.norm(design, 2)) ** 2
        shrink = step * price * (np.ones(width) if costs is None else np.asarray(costs, dtype=float))
        parameters = np.zeros(width + 1) if start is None else np.array([*start.weights, start.bias], dtype=float)
        previous = parameters.copy()
        settled, steps = False, 0
        for steps in range(1, max_steps + 1):
            momentum = parameters + (steps - 1) / (steps + 2) * (parameters - previous)
            candidate = momentum - step * (design.T @ (expit(design @ momentum) - targets)) / rows
            weights = candidate[:width]
            candidate[:width] = np.sign(weights) * np.maximum(np.abs(weights) - shrink, 0.0)
            previous, parameters = parameters, candidate
            if np.max(np.abs(parameters - previous)) <= tolerance * max(1.0, float(np.max(np.abs(parameters[:width]), initial=0.0))):
                settled = True
                break
        return SparseFit(tuple(float(weight) for weight in parameters[:width]), float(parameters[width]), steps, settled)

    def loss(self, columns: np.ndarray, targets: np.ndarray, weights: tuple[float, ...], bias: float) -> float:
        """The mean logistic loss, log(1 + e^z) - target × z with z = columns · weights + bias, without the price."""
        z = columns @ np.asarray(weights, dtype=float) + bias
        return float(np.mean(np.logaddexp(0.0, z) - targets * z))
