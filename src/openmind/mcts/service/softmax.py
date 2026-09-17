import math
from collections.abc import Sequence


def softmax(values: Sequence[float | None], temperature: float) -> tuple[float, ...]:
    """Each value's share, exp(value / temperature) over the sum; a value that is None takes the mean of the others, and
    with none known every share is equal. A temperature of 0 or less raises ValueError."""
    if temperature <= 0.0:
        raise ValueError(f"A softmax needs a temperature above 0, not {temperature}")
    if not values:
        return ()
    known = [value for value in values if value is not None]
    if not known:
        return tuple(1.0 / len(values) for _ in values)
    mean = math.fsum(known) / len(known)
    filled = [mean if value is None else value for value in values]
    top = max(filled)
    weights = [math.exp((value - top) / temperature) for value in filled]
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)
