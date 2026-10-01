import logging
from collections.abc import Mapping, Sequence

from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.training.model.agreement import Agreement
from openmind.training.model.teller_agreement import TellerAgreement

logger = logging.getLogger(__name__)

#: What a heuristic has been measured at against what games paid, over every judging it has been through.
#:
#: The value is the mass it gathered; what it is read against travels beside it, because a share means nothing
#: without the line it is above or below.
JUDGED = "what {model} expected of what happened"

#: The parts of that measure, kept beside it rather than folded in.
#:
#: **Four facts and no score, which is `Agreement`'s own discipline.** A share can be read off these several
#: ways and they mean different things — the mass over the decisions it answered is how well it predicts when
#: it speaks, the same mass over every decision is how well it would play a whole game — and folding them here
#: would make that choice for whoever reads them.
MASS, OFFERED, DECIDED, DECLINED, UNDECIDED, JUDGINGS = (
    "mass", "offered", "decided", "declined", "undecided", "judgings"
)

#: How closely a heuristic's reading of a position tracks a teller's, as a rank agreement from -1 to 1.
#:
#: Kept apart from what the games paid, and never added into it. A teller is a claim with measured reliability
#: and the payoff is the anchor, so a page showing both shows two numbers rather than one blend — and a
#: heuristic that tracks the teller while losing games is a thing somebody should be able to see.
TOLD = "how closely {model} tracks the teller"
TOLD_OF = "positions"


class JudgingRecord:
    """What a judging found, written where it can be read again.

    **The evidence was being computed and thrown away.** Every judging asks every candidate what it would have
    expected of games that were played, and what came back was read once — to pay the signals that vouched for
    it and to retire what had shown nothing — and then dropped. The one number that survived was the worth,
    which is the mass less what ignorance would have expected, so how often a heuristic *spoke at all* was
    gone and with it the difference between a detector that is right where it fires and a policy that is right
    everywhere.

    **Accumulated, because a judging is a handful of games.** A heuristic measured over eighty decisions has
    said little about itself; the same heuristic over eight thousand has. So the counts add and the page reads
    totals, and how many judgings they came from is kept so that nobody mistakes one long look for many.

    The teller's number does not add — it is a rank agreement, not a count — so the latest stands, with how
    many positions it was taken over beside it.

    It keeps nothing: built once, it is given a knowledge base on every call."""

    def remember(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        named: Mapping[str, str],
        agreements: Sequence[Agreement],
    ) -> None:
        """Each heuristic's judging added to what it has shown before.

        `named` is what a judging calls a heuristic against what the store calls it, because the two differ and
        the store's name is what a page will look it up by."""
        for one in agreements:
            model = named.get(one.holder)
            if model is None:
                continue
            variable = JUDGED.format(model=model)
            before = dict((knowledge_base.belief(variable, context_id) or Belief(variable, context_id, 0.0)).tags)
            held = {
                MASS: self._more(before, MASS, one.mass),
                OFFERED: self._more(before, OFFERED, one.offered),
                DECIDED: self._more(before, DECIDED, one.decided),
                DECLINED: self._more(before, DECLINED, one.declined),
                UNDECIDED: self._more(before, UNDECIDED, one.undecided),
                JUDGINGS: self._more(before, JUDGINGS, 1),
            }
            knowledge_base.believe(
                Belief(variable, context_id, held[MASS], tags=tuple(held.items()))
            )

    def told(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        named: Mapping[str, str],
        agreements: Sequence[TellerAgreement],
    ) -> None:
        """What the teller made of each heuristic, the latest standing.

        Not added to the last one: a rank agreement is a correlation and correlations do not sum. What it is
        worth knowing beside it is how many positions it was taken over, since agreement over nine positions
        and over nine thousand are different claims."""
        for one in agreements:
            model = named.get(one.holder)
            if model is None or not one.decided:
                continue
            knowledge_base.believe(
                Belief(
                    TOLD.format(model=model), context_id, one.agreed,
                    tags=((TOLD_OF, one.decided), (DECLINED, one.declined)),
                )
            )

    def _more(self, before: Mapping[str, object], name: str, by: float) -> float:
        """What that tally comes to once this judging is added to it."""
        held = before.get(name, 0)
        return (float(held) if isinstance(held, int | float) else 0.0) + by
