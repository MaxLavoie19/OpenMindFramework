from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatternCondition:
    """One condition of a pattern: the variable of a base at the pattern's index shifted by steps, compared (== or !=)
    with a value, written as source (`me`, `'N'`, `None`), or with the variable another condition reads, as the index of
    that condition."""

    base: str
    steps: tuple[int, ...]
    relation: str
    value: str | None = None
    other_condition: int | None = None
