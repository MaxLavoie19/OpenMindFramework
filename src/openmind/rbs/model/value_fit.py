from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueFit:
    """One price of a sweep: how many terms kept a weight, the steps taken, whether the weights settled, and the mean
    logistic loss on the training rows and on the held-out rows (None without held-out rows)."""

    price: float
    terms_kept: int
    steps: int
    settled: bool
    training_loss: float
    held_out_loss: float | None
