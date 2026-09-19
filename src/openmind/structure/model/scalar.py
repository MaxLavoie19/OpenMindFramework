from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Scalar:
    """One value, such as the player to act or a score. Rules read it as the value itself."""

    value: Value

    def with_value(self, value: Value) -> "Scalar":
        return Scalar(value)
