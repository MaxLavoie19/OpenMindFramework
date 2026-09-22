from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from openmind.inference.model.example import Example
from openmind.rule.model.literal import Literal


@dataclass(frozen=True, slots=True)
class CaseIndex:
    """Which cases were read as holding which reading, so a clause is only asked of the cases that could say yes.

    A clause holds of a case only where every one of its settled readings — the ones with nothing left open — is
    among that case's own. So a clause carrying the reading that what stands at the source is of one particular
    kind cannot possibly hold of a case that read something else there, and asking is a waste of the asking.

    That matters because of which questions are expensive. A clause generalised too far is caught by nearly the
    first case that contradicts it, and costs a handful of questions. A clause that is *right* contradicts nothing,
    so finding that out means asking every case there is — and it is the right ones a learner keeps, so the right
    ones are what it spends its time on. Narrowing first turns that walk over everything into a walk over the few
    cases that share a reading with the clause.

    The rarest reading is the one chosen, because it is the one that rules out the most. Its cases are a superset
    of the cases the clause can hold of, never a subset, so nothing is hidden by looking there: the caller still
    asks properly of each one, it is simply asked of far fewer.

    A clause with no settled reading at all — everything in it open — narrows to nothing, and every case is
    offered. There is no loss in that. Such a clause says almost nothing, so it is contradicted at once."""

    cases: tuple[Example, ...]

    #: Each reading against the cases that were read as having it, worked out once when the index is made.
    holders: Mapping[Literal, tuple[Example, ...]] = field(
        init=False, compare=False, hash=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        gathered: dict[Literal, list[Example]] = {}
        for case in self.cases:
            for literal in case.literals:
                gathered.setdefault(literal, []).append(case)
        object.__setattr__(self, "holders", {literal: tuple(found) for literal, found in gathered.items()})

    def narrowed(self, literals: Sequence[Literal]) -> tuple[Example, ...]:
        """The cases worth asking about a clause whose settled readings are those.

        Every case the clause can hold of is in here. Cases that cannot are mostly not."""
        fewest: tuple[Example, ...] | None = None
        for literal in literals:
            holding = self.holders.get(literal, ())
            if fewest is None or len(holding) < len(fewest):
                fewest = holding
                if not fewest:
                    break
        return self.cases if fewest is None else fewest
