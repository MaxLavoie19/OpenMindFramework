import logging
from collections.abc import Iterable, Mapping
from math import inf, lgamma, log, log2

logger = logging.getLogger(__name__)

#: Rissanen's constant, which makes the universal code for a whole number sum to one over all of them.
#:
#: It is not a tuning number and nobody chose it. It is what normalises `log* i` into a probability
#: distribution, so that the code lengths it gives are code lengths of something rather than scores.
KRAFT = 2.865064


class Information:
    """How much is said, measured in bits.

    **Here because it kept being written out.** Shannon's formula appeared verbatim in two places — the syntax
    learner deciding how many sorts a notation has, and the coupling learner deciding whether a symbol says
    anything about a part — and a third place priced rules without sharing either's notion of a bit. Three
    callers, three copies, one idea. That is the mark of something that should have a name.

    **Nothing here is about games.** How many bits a frequency carries, how many it takes to say which k of n,
    how many to say a number nobody bounded: none of it knows what a board is. That is what makes it a theory
    rather than a service — it is true before any game is declared and stays true after.

    It keeps nothing: built once, it is given the counts on every call."""

    def entropy(self, counted: Mapping[object, float] | Iterable[float]) -> float:
        """Bits per thing, over those frequencies: how surprised one is on average by which it turns out to be.

        Nought where everything is the same thing — nothing is learned by being told which, because there was
        never a question. Highest where they are all equally likely, which is the case with nothing to go on."""
        held = list(counted.values() if isinstance(counted, Mapping) else counted)
        total = sum(held)
        if total <= 0:
            return 0.0
        return -sum(one / total * log2(one / total) for one in held if one)

    def told(self, before: Mapping[object, float], after: Mapping[object, float]) -> float:
        """How many bits knowing something saves: what a thing cost to say before, less what it costs now.

        Never negative in principle, since knowing more cannot cost more — but a measurement over few enough
        sightings can come out that way, and saying so is better than clamping it silently."""
        return self.entropy(before) - self.entropy(after)

    def counting(self, number: int) -> float:
        """Bits to say a whole number nobody put a bound on.

        Rissanen's universal code: say the number, then say how long that was, then how long *that* was, until
        there is nothing left to say. It is what one uses where a bound would be invented — how many constraints
        a game needs is not a number anybody may assume."""
        if number < 1:
            return log2(KRAFT)
        found, held = log2(KRAFT), float(number)
        while held > 1.0:
            held = log2(held)
            if held > 0.0:
                found += held
        return found

    def choosing(self, among: int, taken: int) -> float:
        """Bits to say which `taken` of `among`, worked out through log-gammas so that large boards do not need
        large integers."""
        if taken < 0 or taken > among:
            return inf
        return (lgamma(among + 1) - lgamma(taken + 1) - lgamma(among - taken + 1)) / log(2)

    def naming(self, among: int) -> float:
        """Bits to point out one of that many.

        The plainest measure there is, and the one that makes a theory pay for itself: the fewer a set of rules
        leaves standing, the less it takes to say which of them happened."""
        return log2(among) if among > 0 else inf

    def ordering(self, held: int) -> float:
        """Bits in the number of orders that many things come in.

        Credited rather than charged, where the things have no order: a set of K could have been written down K!
        ways and they all mean the same, so a code charging for the order charges for something carrying
        nothing."""
        return lgamma(held + 1) / log(2)
