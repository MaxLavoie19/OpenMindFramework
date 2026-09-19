from typing import Self

from openmind.structure.model.data_model import DataModel
from openmind.structure.model.value import Value
from openmind.world.model.state import State


class StateBuilder:
    """Collects named data models into a state; a plain value becomes a Scalar."""

    def __init__(self) -> None:
        self._models: dict[str, DataModel | Value] = {}

    def with_model(self, name: str, model: DataModel | Value) -> Self:
        if name in self._models:
            raise ValueError(f"State model {name!r} is already set")
        self._models[name] = model
        return self

    def build(self) -> State:
        return State.of(**self._models)
