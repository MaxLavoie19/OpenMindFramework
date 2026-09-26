from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Shape:
    """One layout a notation comes in: which sort stands at each place.

    A notation belongs to exactly one shape, and a shape fixes its length. That is what makes a place in it worth
    measuring, where a place in the raw string was not: the same character says one thing in the opening place of
    one shape and another thing in the opening place of another, and only the shape tells them apart."""

    sorts: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.sorts)

    @property
    def readable(self) -> str:
        return " ".join(f"[{one}]" for one in self.sorts)
