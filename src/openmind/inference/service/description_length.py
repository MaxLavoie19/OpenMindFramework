import logging
from collections.abc import Sequence

from openmind.inference.service.information import Information
from openmind.rule.model.clause import Clause

logger = logging.getLogger(__name__)


class DescriptionLength:
    """What a set of constraints costs to say, and what it leaves to be said.

    **A code and not a penalty, which is the whole change.** What we had was `1 + length²`: a made-up number that
    grows with a constraint's length. It sorts, so it looked like a criterion, and the literature's name for that
    is short-circuiting the principle — a unit cost avoids making the decisions a code has to make, and then the
    answer is a fact about the formula rather than about the evidence.

    **And it had the sign wrong.** Under a real code, saying which k of C conditions a body uses costs
    `log₂C(C, k)`, whose *marginal* cost per further condition falls as the body grows — the fifth condition is
    cheaper than the third, because there are fewer ways left to choose it. Under `1 + k²` the marginal cost
    rises. So the quadratic systematically preferred short constraints, and a short constraint here is an
    over-general one, which is the mistake this project calls unrecoverable: a move OMF will never make and will
    never hear it could have.

    **What the data costs is the thing nothing priced at all.** The constraints are not judged against a corpus —
    there is no corpus. They are judged against the moves the game has shown to be legal, and what those cost to
    say is `log₂` of how many candidates survive the constraints: with everything refused but forty, naming one
    of them takes 5.3 bits; with nothing refused at all, 13.8. **So tightening the constraints is paid for, move
    by move, without any generator and without a single negative example.** That is what makes "fourteen rules
    leaving some of it unaccounted for" a thing that can be preferred or rejected rather than a thing that cannot
    be said.

    **The asymmetry we care about falls out rather than being weighted in.** Letting one more illegal candidate
    survive moves the naming cost from log₂40 to log₂41 — under four hundredths of a bit. Wrongly refusing a
    known-legal one makes it an exception, which must be named among all fourteen thousand: about sixteen bits.
    Four hundred and fifty to one, out of the encoding. Nothing here was tuned to produce it, which is why it can
    be trusted in a position nobody has looked at.

    It keeps nothing but the size of the vocabulary a condition is drawn from."""

    def __init__(self, conditions: int = 1, information: Information | None = None) -> None:
        # What bits are measured in. Kept apart from this because how many bits a thing costs is not a fact
        # about constraints: the same counting prices a notation's sorts and a symbol's worth.
        self._information = Information() if information is None else information
        #: How many conditions there are to choose from — the vocabulary a body is written in.
        #:
        #: It shifts every constraint's price by a constant, and so shifts how many are worth buying. The honest
        #: value is the number of distinct readings in the pool being priced, which the caller knows and this
        #: does not.
        self._conditions = max(1, conditions)

    def counting(self, number: int) -> float:
        """Bits to say a whole number nobody put a bound on."""
        return self._information.counting(number)

    def choosing(self, among: int, taken: int) -> float:
        """Bits to say which `taken` of `among`."""
        return self._information.choosing(among, taken)

    def saying(self, clause: Clause) -> float:
        """What one constraint costs: how many conditions it has, and which ones they are."""
        held = len(clause.body)
        return self.counting(held) + self.choosing(self._conditions, held)

    def theory(self, clauses: Sequence[Clause]) -> float:
        """What a set of them costs, said as a set.

        **The `- log₂(K!)` is earned and not a discount.** A set of K constraints could have been written down in
        K! orders and they all mean the same thing, so a code that charges for the order is charging for
        something that carries nothing. It comes to about twenty-two bits at ten constraints and sixty-one at
        twenty — which is the difference between a set that pays for itself and one that does not."""
        held = len(clauses)
        if not held:
            return self.counting(0)
        return self.counting(held) - self._factorial(held) + sum(self.saying(one) for one in clauses)

    def naming(self, survivors: int, legal: int) -> float:
        """What it costs to point out the moves the game allows, among everything the constraints let through.

        This is the whole of the data. The constraints are never shown a corpus and never shown a negative; what
        they are shown is that these particular candidates are legal, and the better the constraints, the less it
        takes to say which ones.

        Where nothing survives and something was legal, there is nothing to say it with — the constraints refuse
        a move the game allows, which is not a price but a contradiction."""
        if legal <= 0:
            return 0.0
        if survivors < legal:
            return float("inf")
        return legal * self._information.naming(survivors)

    def whole(self, clauses: Sequence[Clause], seen: Sequence[tuple[int, int]]) -> float:
        """The description length of those constraints against those positions.

        `seen` is one pair per position: how many candidates survive the constraints there, and how many of them
        the game actually allows. Both come from the coverage the learner already computes."""
        return self.theory(clauses) + sum(self.naming(survivors, legal) for survivors, legal in seen)

    def _factorial(self, held: int) -> float:
        """Bits in the number of orders K things come in."""
        return self._information.ordering(held)
