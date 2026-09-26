from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Correspondence:
    """One thing a whole family of a notation's symbols says, said once.

    Eight digits naming eight rows are eight couplings and a lookup table: nothing in them is about *digits*, and
    a ninth row would be unreadable however many of the eight had been seen. Said as a correspondence they are
    one thing — the symbols run in an order, the values run in an order, and one runs against the other — which
    covers the ninth without having met it. That is the difference between a rule and a list, in the notation
    rather than in the rules.

    `symbols` and `values` are the two orders, paired by place. `turned` says they run against each other, which
    is how a notation counting ranks from one side reads onto a grid counting rows from the other.

    Nothing about numbers is assumed. The orders are whatever order the two sets were found in, so letters
    running onto columns are the same thing as digits running onto rows, and a notation whose symbols are words
    would be no different."""

    part: str
    symbols: tuple[str, ...]
    #: Numbers, and only numbers. Two things run in step when there is a step to run in, and the order of a set of
    #: names is the order somebody spelled them in. Three piece kinds whose letters happen to sort as their names
    #: do — B, N, R against bishop, knight, rook — would otherwise be read as a correspondence, which is a fact
    #: about English and not about chess.
    values: tuple[int, ...]
    turned: bool = False
    telling: float = 0.0

    def says(self, symbol: str) -> Value | None:
        """What that symbol says of the part, including for a symbol it has never met.

        **Reaching past what was seen is the whole point.** A paired list of eight digits and eight rows is still
        a list; what makes it a rule is that a ninth digit reads as a ninth row on a board that has one. So the
        pairing is kept as a step from the first symbol to the first value, and a symbol beyond the ones seen is
        carried on by the same step rather than refused.

        It only reaches where the symbols themselves run in steps — single characters do, and words do not — and
        it says nothing where they do not."""
        if symbol in self.symbols:
            where = self.symbols.index(symbol)
            return self.values[len(self.values) - 1 - where] if self.turned else self.values[where]
        if len(symbol) != 1 or len(self.symbols[0]) != 1:
            return None
        along = ord(symbol) - ord(self.symbols[0])
        first = self.values[-1] if self.turned else self.values[0]
        return first - along if self.turned else first + along

    def reaches(self, symbol: str, value: Value) -> bool:
        """Whether the correspondence would have said that, which is how it is checked against a sighting."""
        return self.says(symbol) == value

    @property
    def readable(self) -> str:
        how = "against" if self.turned else "with"
        return (
            f"{self.symbols[0]!r}..{self.symbols[-1]!r} say {self.part}, running {how} "
            f"{self.values[0]!r}..{self.values[-1]!r} ({self.telling:.2f} bits)"
        )
