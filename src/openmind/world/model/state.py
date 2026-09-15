from dataclasses import dataclass, field

from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class State:
    """Named variables with their values, sorted by name. Its hash is worked out once; a copy sent to another process
    carries only its variables, since that process hashes names its own way."""

    variables: tuple[tuple[str, Value], ...]
    _hash: int | None = field(default=None, init=False, repr=False, compare=False)

    def __hash__(self) -> int:
        cached = self._hash
        if cached is None:
            cached = hash(self.variables)
            object.__setattr__(self, "_hash", cached)
        return cached

    def __reduce__(self) -> tuple[type["State"], tuple[tuple[tuple[str, Value], ...]]]:
        return (State, (self.variables,))
