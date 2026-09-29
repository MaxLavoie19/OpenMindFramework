import logging

from openmind.inference.model.expression import Expression
from openmind.inference.service.information import Information

logger = logging.getLogger(__name__)


class RulePrice:
    """What a rule costs to vouch for: the bits it takes to write down, over the vocabulary it is written in.

    **A code, not a penalty, and this project has already paid for the difference.** The first attempt priced a
    body of k conditions at `1 + k²`. `doc/learner-practices.md` records why it was struck: under a real code,
    saying which k of C conditions a body uses costs `log₂C(C, k)`, whose marginal cost per further condition
    **falls** as the body grows, because there are fewer ways left to choose it. Under `1 + k²` it **rises**. So
    the quadratic systematically preferred short rules, and a short rule is an over-general one.

    **What this gives is longer rules being dearer without anybody choosing how much dearer.** That is the whole
    of what "more clauses are more expensive, so they exist but are rarer" should mean: a signal can buy a long
    rule, it just has to have saved for it.

    **The price is in bits, and so is what a signal earns.** `RuleBudget`'s allowance is the exchange rate, and
    it is the caller's to set — a larger one buys more rules and a smaller one fewer. Nothing here decides how
    many bits a heuristic's worth is worth; this only says what a rule costs to say.

    It keeps nothing but the counting, which is the same counting that prices a notation's sorts and a
    constraint's body."""

    def __init__(self, information: Information | None = None) -> None:
        self._information = Information() if information is None else information

    def priced(self, expression: Expression, vocabulary: int) -> float:
        """What that expression costs: how many conditions and operations it combines, and which ones they are.

        `vocabulary` is how many readings the term was drawn from — the pool the search generated over, which
        the caller knows and this does not. It shifts every price by roughly a constant and so shifts how many
        rules are worth buying; it is not a knob, it is a fact about the search that produced the candidates.

        A vocabulary smaller than what the expression uses is not an error worth raising — a pool can shrink
        between a search and a judging — so the choosing is taken over at least what was chosen."""
        clauses = max(0, expression.clauses)
        among = max(vocabulary, clauses, 1)
        return self._information.counting(clauses) + self._information.choosing(among, clauses)
