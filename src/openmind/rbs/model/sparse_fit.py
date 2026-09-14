from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SparseFit:
    """Weights and a bias fitted at an L1 price, the steps taken, and whether the weights settled before the step limit;
    a weight of exactly 0 drops its term."""

    weights: tuple[float, ...]
    bias: float
    steps: int
    settled: bool
