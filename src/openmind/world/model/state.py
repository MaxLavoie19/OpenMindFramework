from dataclasses import dataclass, field

from openmind.structure.model.data_model import DataModel
from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class State:
    """Named data models, sorted by name: `cell` a Grid, `turn` a Scalar, `payoff` a Map. Its hash is worked out once; a
    copy sent to another process carries only its models, since that process hashes names its own way."""

    models: tuple[tuple[str, DataModel], ...]
    _hash: int | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def of(**models: DataModel | Value) -> "State":
        """A state from its models by name; a plain value becomes a Scalar."""
        return State(
            tuple(sorted(((name, _model(model)) for name, model in models.items()), key=lambda item: item[0]))
        )

    def __hash__(self) -> int:
        cached = self._hash
        if cached is None:
            cached = hash(self.models)
            object.__setattr__(self, "_hash", cached)
        return cached

    def __reduce__(self) -> tuple[type["State"], tuple[tuple[tuple[str, DataModel], ...]]]:
        return (State, (self.models,))

    def model(self, name: str) -> DataModel:
        for held, model in self.models:
            if held == name:
                return model
        raise KeyError(f"Unknown state model: {name!r}")

    def has(self, name: str) -> bool:
        return any(held == name for held, _ in self.models)

    def value(self, name: str) -> Value:
        """A scalar's value."""
        model = self.model(name)
        if not isinstance(model, Scalar):
            raise TypeError(f"{name} is a {type(model).__name__}, not a Scalar")
        return model.value

    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.models)

    def with_model(self, name: str, model: DataModel | Value) -> "State":
        """The state with that model set, added where it had none."""
        kept = {held: held_model for held, held_model in self.models}
        kept[name] = _model(model)
        return State(tuple(sorted(kept.items(), key=lambda item: item[0])))


def _model(model: DataModel | Value) -> DataModel:
    return model if isinstance(model, Scalar | List | Grid | Map) else Scalar(model)
