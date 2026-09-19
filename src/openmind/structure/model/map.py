from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Map:
    """Values by key, such as each player's payoff. Its items are kept sorted by key, so two maps holding the same items
    are equal. Every change gives a new map."""

    items: tuple[tuple[Value, Value], ...] = ()

    @staticmethod
    def of(values: Mapping[Value, Value]) -> "Map":
        return Map(tuple(sorted(values.items(), key=lambda item: repr(item[0]))))

    def __getitem__(self, key: Value) -> Value:
        for held, value in self.items:
            if held == key:
                return value
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return any(held == key for held, _ in self.items)

    def __iter__(self) -> Iterator[Value]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.items)

    def get(self, key: Value, default: Value = None) -> Value:
        return self[key] if key in self else default

    def keys(self) -> tuple[Value, ...]:
        return tuple(key for key, _ in self.items)

    def values(self) -> tuple[Value, ...]:
        return tuple(value for _, value in self.items)

    def with_item(self, key: Value, value: Value) -> "Map":
        return Map.of({**dict(self.items), key: value})
