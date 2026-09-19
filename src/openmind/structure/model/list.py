from collections.abc import Iterator
from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class List:
    """An ordered sequence of values, such as a hand of cards or a history. Every change gives a new list."""

    items: tuple[Value, ...] = ()

    def __getitem__(self, index: int) -> Value:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Value]:
        return iter(self.items)

    def __contains__(self, item: object) -> bool:
        return item in self.items

    def appended(self, item: Value) -> "List":
        return List((*self.items, item))

    def removed(self, index: int) -> "List":
        return List(self.items[:index] + self.items[index + 1 :])

    def replaced(self, index: int, item: Value) -> "List":
        return List(self.items[:index] + (item,) + self.items[index + 1 :])

    def count(self, item: Value) -> int:
        return self.items.count(item)
