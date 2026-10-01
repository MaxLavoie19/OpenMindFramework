from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeuristicStanding:
    """One heuristic as a page shows it: what it has been measured at, by each measure, and how much it rests on.

    **Each measure stands on its own and none is folded into another.** What games paid is the anchor; what a
    teller makes of a position is a claim with measured reliability beside it. A heuristic may do well on one
    and badly on the other, and that is the thing worth seeing — a page showing one blended number would hide
    exactly the case somebody wants to look at.

    `worth` is what it expected of what was played less what a heuristic with no opinion would have expected,
    added up over every judging. Above nought it saw something, at nought it knows nothing, below it is
    actively wrong about what wins. `mass` and `offered` are the two halves of that, kept apart because the
    same difference means different things at different coverage.

    `decided` is how many decisions it answered, `declined` where it knew nothing, and `undecided` where its
    opinion separated nothing. Coverage is read beside the score and never folded into it: a detector held to a
    policy's coverage is a detector marked down for being a detector.

    `tracks` is how closely its reading of a position follows the teller's, from -1 to 1, over `told`
    positions; `None` where the teller has not been asked about it. It is the latest and not a total, because a
    rank agreement is a correlation and correlations do not sum.

    `games`, `wins`, `draws` and `losses` are what it came to where it was actually drawn to play, which most
    candidates never are — a heuristic is measured over games somebody else played, which is what makes a large
    pool affordable.

    `retired` is whether it has been taken out of the pool for having shown enough, and shown nothing."""

    name: str
    worth: float = 0.0
    mass: float = 0.0
    offered: float = 0.0
    decided: int = 0
    declined: int = 0
    undecided: int = 0
    judgings: int = 0
    tracks: float | None = None
    told: int = 0
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    vouched: tuple[str, ...] = ()
    retired: bool = False
    #: Each of its rules, with what every signal made of it.
    #:
    #: **The only account of why a rule is in a heuristic.** The signals rate every candidate a fit kept and
    #: spend on the ones they want; what they thought was used to decide and then dropped, so a heuristic could
    #: say who had backed it and never what any of them made of any particular rule.
    #:
    #: Every signal and not only the buyers, because a signal rating a rule at nearly nothing says as much
    #: about that rule as one that paid for it — most of all where the two disagree, which a table of buyers
    #: cannot show at all.
    rated: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()

    @property
    def asked(self) -> int:
        """Every decision it was put to, answered or not."""
        return self.decided + self.declined + self.undecided

    @property
    def coverage(self) -> float:
        """The share of decisions it had an opinion about, or nought where it was never asked."""
        return self.decided / self.asked if self.asked else 0.0

    @property
    def speaks(self) -> str:
        """How often it says anything at all, for a page to show beside what it says."""
        return f"{self.decided} of {self.asked}" if self.asked else "never asked"
