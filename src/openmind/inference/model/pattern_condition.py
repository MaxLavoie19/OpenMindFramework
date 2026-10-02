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
    #: A whole test of its own, used in place of the base and the comparison: an idiom.
    #:
    #: **A term is only short in the words it is written in.** A fork is two pieces of theirs that can be taken,
    #: which is a dozen conditions over cells and two over idioms — and every search in the literature is
    #: capped at four to six conditions, so the long way round is not merely slower but out of reach. What
    #: closes that is not a bigger search but a bigger vocabulary: a term that earned its keep is named, and
    #: the next assembly may use it as one condition.
    #:
    #: `name` is what it is called where it is read; `says` is what it reads. A condition carrying one is
    #: written out as it stands, so nothing downstream needs to know an idiom from a cell.
    says: str | None = None
    name: str | None = None
