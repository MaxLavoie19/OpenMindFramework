from dataclasses import dataclass, field

from openmind.rule.model.clause import Clause


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One thing that might be a rule, and what it came to when it was tried.

    **The unit of work, and the unit of sharing.** What a body refuses does not depend on which case somebody was
    trying to account for when they wrote it down, so it is worked out once and serves every case it covers. That
    is the whole of why many workers can search one space without doing each other's work again: they do not
    share a frontier, they share answers.

    **Places and not cases.** `refusing` holds where in the pool each case sits, because a place is a number and
    a case is a hundred and fifty readings. Nothing about a case ever crosses between processes; the pool is read
    where it is used, and the places mean the same thing on both sides because the pool is the same list.

    **Both facts are kept, and they are not the same kind of fact.** That a body refuses nothing is a *prune*: no
    body containing it can refuse anything either, since conditions only ever narrow. That a body slips is not a
    prune at all — every subset of it slips too, and a search by increasing size has already seen every subset —
    it is a *memory*, so that the next worker to reach this body from another case does not test it again."""

    #: The rule itself. Its head is always `refused`; the body is what varies.
    clause: Clause
    #: Where in the pool each case it refuses sits.
    refusing: frozenset[int] = frozenset()
    #: Whether it turns away anything the game allows. A hypothesis that slips is kept, not discarded: keeping it
    #: is what stops it being tried again, which is the only thing it is now good for.
    slips: bool = False

    #: How many conditions it has, worked out once so a table can be ordered without opening the clause.
    size: int = field(init=False, compare=False, hash=False, repr=False, default=0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", len(self.clause.body))

    @property
    def key(self) -> frozenset:
        """What makes two of them the same hypothesis, said cheaply.

        A conjunction is a set: the same conditions in another order are the same rule, and a body reached from
        two different cases is one entry rather than two. Said as the set of its conditions — never by asking
        whether one says everything another says, which is the right question and costs far too much to ask of
        every candidate that arrives."""
        return frozenset(self.clause.body)

    @property
    def useful(self) -> bool:
        """Whether it is worth keeping for anything but the memory that it failed.

        What may be chosen from is the rules that refuse something and turn nothing away wrongly. Everything else
        is kept only so it is not tried again, and a table that offers it all to whatever picks the final set is
        the arrangement that makes picking intractable."""
        return bool(self.refusing) and not self.slips
