from openmind.structure.model.value import Value


class EvenErrorModel:
    """The error model of a mechanism that has none of its own: right as often as its accuracy says, and when wrong,
    any of the other values it could have said, evenly."""

    def likelihood(self, said: Value, true: Value, accuracy: float, candidates: int) -> float:
        if said == true:
            return accuracy
        return (1.0 - accuracy) / max(1, candidates - 1)
