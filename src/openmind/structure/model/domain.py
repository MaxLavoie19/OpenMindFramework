from dataclasses import dataclass, replace

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Numbers:
    """A set of numbers a parameter may range over, which need not be listable.

    **A game names one of these rather than spelling out a range.** Chess declares a move as two whole numbers,
    signed, and says nothing about eight — so *you may not move off the grid* is a rule OMF has to find, where a
    declared range of minus seven to seven would have handed it over. The same declaration describes checkers,
    shogi or amazons, and what separates those games is entirely what gets learned from them.

    `least` and `most` are where it is bounded, and None where it is not. An unbounded set cannot be listed, so
    nothing may enumerate one without narrowing it first — and a narrowing is only honest where it can be shown
    that nothing outside it could have been anything but refused, for the same reason as something inside."""

    whole: bool = True
    least: float | None = None
    most: float | None = None

    @property
    def listable(self) -> bool:
        """Whether every value of it can be handed out: bounded both ways, and whole."""
        return self.whole and self.least is not None and self.most is not None

    def within(self, least: float, most: float) -> "Numbers":
        """The same numbers, no wider than that.

        Narrowing and never widening: where it was already bounded more tightly, the tighter bound stands."""
        return replace(
            self,
            least=least if self.least is None else max(self.least, least),
            most=most if self.most is None else min(self.most, most),
        )

    @property
    def domain(self) -> tuple[Value, ...]:
        """Every value of it, where they can be listed, and nothing where they cannot.

        Empty is not "no values" — it is "not listable", and a caller that enumerates without asking `listable`
        first will silently offer no candidates rather than fail. Whoever narrows is who knows it is safe to."""
        if not self.listable:
            return ()
        return tuple(range(int(self.least), int(self.most) + 1))

    @property
    def readable(self) -> str:
        held = "whole numbers" if self.whole else "numbers"
        if self.least is None and self.most is None:
            return held
        if self.least is None:
            return f"{held} up to {self.most:g}"
        if self.most is None:
            return f"{held} from {self.least:g}"
        return f"{held} from {self.least:g} to {self.most:g}"


class Domain:
    """The sets OMF knows how to range over, so that a game names one instead of describing it.

    Positive begins above zero where the numbers are whole, since there is a next one to begin at. Among the
    reals there is no next number above zero, so the bound is zero itself and it is taken in — a difference
    between the two that a bound of one number cannot hide, and which is better said here than discovered."""

    WHOLE = Numbers(whole=True)
    WHOLE_POSITIVE = Numbers(whole=True, least=1)
    WHOLE_NEGATIVE = Numbers(whole=True, most=-1)
    REAL = Numbers(whole=False)
    REAL_POSITIVE = Numbers(whole=False, least=0)
    REAL_NEGATIVE = Numbers(whole=False, most=0)
