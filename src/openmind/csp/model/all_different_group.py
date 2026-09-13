from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AllDifferentGroup:
    """Parameters that must all take different values."""

    variables: tuple[str, ...]
