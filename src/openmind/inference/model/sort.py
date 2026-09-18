from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Sort:
    """A kind of thing a term can be, by name: a truth value, a number, or a kind a theory or a game declares."""

    name: str


@dataclass(frozen=True, slots=True)
class SetSort(Sort):
    """The sort of the sets whose members are of the element sort."""

    element: Sort = Sort("")
