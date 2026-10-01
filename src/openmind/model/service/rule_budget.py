import logging
from collections.abc import Mapping, Sequence

from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

#: What a signal has to spend, by signal.
BUDGET = "budget of {signal}"

#: The tags a budget keeps its history under.
VOUCHED = "vouched"
EARNED = "earned"

#: How much of a heuristic one signal backed, and the tag every such belief carries so that all of a
#: heuristic's backers can be found again when its worth is finally known.
VOUCHING = "{signal} vouched for {heuristic}"
VOUCHED_FOR = "vouched for"

#: What one signal made of one rule of one heuristic, and the two tags that find it again.
#:
#: **The only account of *why* a rule is in a heuristic.** A signal rates every candidate and spends on the
#: ones it wants; the ratings were used to decide and then dropped, so a heuristic said who had backed it and
#: never what any of them had thought of any particular rule.
#:
#: The latest and never a total: a ponder rates afresh over its own rows, so what a signal thought last time
#: is about a different fit and adding the two would be adding opinions of different things.
RATING = "{signal} rated {rule} of {heuristic}"
RATED_FOR = "rated for"
RATED_RULE = "rated rule"
RATED_BY = "rated by"

#: What every signal is given each round, and the floor decay may not take a budget below.
#:
#: **An income rather than a balance, which is the difference between never shut out and never spent.** A
#: floor on what a signal *holds* is an infinite well: anything priced at or under it can be bought for ever,
#: however wrong the signal has been. A floor on what it *receives* leaves a signal that has spent everything
#: broke until the next round, and never broke for good — so a run of bad luck costs time rather than a
#: voice. Written the other way first, and the tests hung buying the same rule for ever.
ALLOWANCE = 1.0


class RuleBudget:
    """What each signal has earned, and what it has spent vouching for rules.

    **A signal is a way of proposing that a rule earns its place**, and its budget is what the heuristics it
    vouched for turned out to be worth. It spends that budget to vouch for more.

    **Why a budget and not a weight.** Weighted by how good its heuristics have been, the leading signal wins
    nearly every draw and the rest stop contributing at all — so a signal that is right about a narrow class
    of rules never gets to say so. A signal that earns little buys rarely and saves in between, which keeps
    its voice in proportion to what it has earned without ever silencing it.

    **A bad vouch decays the budget rather than charging it.** Decay is gentler than a fine and it compounds,
    so a signal that keeps being wrong buys less and less; and it never reaches nought, so no signal is
    permanently shut out.

    Every number here is a belief, so what a run learns about which signals are worth listening to outlives
    the run that learned it."""

    def __init__(self, allowance: float = ALLOWANCE) -> None:
        self._allowance = allowance

    def held(self, knowledge_base: KnowledgeBase, context_id: str, signal: str) -> float:
        """What that signal has left to spend, which is nought where it has spent everything."""
        belief = knowledge_base.belief(BUDGET.format(signal=signal), context_id)
        if belief is None or not isinstance(belief.value, int | float):
            return 0.0
        return max(0.0, float(belief.value))  # type: ignore[arg-type]

    def allowed(self, knowledge_base: KnowledgeBase, context_id: str, signals: Sequence[str]) -> None:
        """Every signal's income for the round, which is what keeps a broke one from being broke for good.

        **This is the knob that decides how much comes through.** A larger allowance buys more rules and a
        smaller one fewer, and it is a budget in this project's sense — the caller's to set, never a number
        chosen inside a service."""
        for signal in signals:
            self._keep(knowledge_base, context_id, signal, self.held(knowledge_base, context_id, signal) + self._allowance, EARNED)

    def afford(self, knowledge_base: KnowledgeBase, context_id: str, signal: str, price: float) -> bool:
        """Whether that signal can pay that much for a rule right now."""
        return self.held(knowledge_base, context_id, signal) >= price

    def vouched(self, knowledge_base: KnowledgeBase, context_id: str, signal: str, price: float) -> bool:
        """That signal staking the price on a rule: spent where it can be afforded, and nothing where it
        cannot.

        **What is bought is the rule's admission and not its tenure.** A rule a signal paid for enters the
        ruleset and then stands on its own record, so this says only that somebody was willing to back it."""
        held = self.held(knowledge_base, context_id, signal)
        if held < price:
            return False
        self._keep(knowledge_base, context_id, signal, held - price, VOUCHED)
        return True

    def earned(
        self, knowledge_base: KnowledgeBase, context_id: str, signals: Mapping[str, float], worth: float
    ) -> None:
        """What the heuristics a signal vouched for turned out to be worth, paid to it or decayed from it.

        `signals` is how much of the ruleset each signal vouched for, as a share, so a signal that backed one
        rule of ten is paid for one rule of ten. The worth is the ruleset's, because what a heuristic is worth
        is a fact about the whole of it — which is the reason a rule's own standing is read by leaving it out
        rather than by asking what it did alone.

        **Worth below nothing decays rather than subtracting.** A heuristic that did worse than knowing
        nothing says its backers were wrong, and decay makes that cost compound without ever emptying a purse.
        """
        for signal, share in signals.items():
            held = self.held(knowledge_base, context_id, signal)
            if worth >= 0:
                moved = held + worth * share
            else:
                # Decayed, and never below the allowance by decaying — being wrong costs what a signal has
                # built up, not its ability to try again next round.
                decayed = held * (1.0 - min(1.0, -worth * share))
                moved = max(decayed, min(held, self._allowance))
            self._keep(knowledge_base, context_id, signal, moved, EARNED)
        logger.debug(
            "%s %.3f among %s",
            "Paid" if worth >= 0 else "Decayed by",
            abs(worth),
            ", ".join(f"{one} for {two:.2f} of it" for one, two in signals.items()),
        )

    def vouching(
        self, knowledge_base: KnowledgeBase, context_id: str, heuristic: str, shares: Mapping[str, float]
    ) -> None:
        """Who backed how much of that heuristic, written down where its worth will later be measured.

        **A heuristic is vouched for long before anybody knows what it is worth.** The signals spend during a
        ponder; what the ruleset they bought turns out to be worth is only known once it has played and been
        judged, which is somewhere else and much later. So the debt is recorded against the heuristic's name
        and settled when the number arrives — without this, a signal could never be paid for being right."""
        for signal, share in shares.items():
            knowledge_base.believe(
                Belief(
                    VOUCHING.format(signal=signal, heuristic=heuristic),
                    context_id,
                    float(share),
                    tags=((VOUCHED_FOR, heuristic),),
                )
            )

    def rating(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        heuristic: str,
        ratings: Mapping[str, Mapping[str, float]],
    ) -> None:
        """What every signal made of each rule of that heuristic, written where a page can read it again.

        **Every signal and not only the buyers.** A signal rating a rule at nearly nothing says as much about
        that rule as one that paid for it — more, where the two disagree — and a table with only the buyers in
        it cannot show a disagreement at all.

        Written against the heuristic's name the way the vouching is, and settled nowhere: this is not a debt,
        it is a record of what was thought."""
        for rule, held in ratings.items():
            for signal, rate in held.items():
                knowledge_base.believe(
                    Belief(
                        RATING.format(signal=signal, rule=rule, heuristic=heuristic),
                        context_id,
                        float(rate),
                        tags=((RATED_FOR, heuristic), (RATED_RULE, rule), (RATED_BY, signal)),
                    )
                )

    def ratings(
        self, knowledge_base: KnowledgeBase, context_id: str, heuristic: str
    ) -> Mapping[str, Mapping[str, float]]:
        """What every signal made of each of its rules, by the rule's own name, or nothing where none said."""
        found: dict[str, dict[str, float]] = {}
        for belief in knowledge_base.beliefs(context_id, tags=((RATED_FOR, heuristic),)):
            if not isinstance(belief.value, int | float):
                continue
            tags = dict(belief.tags)
            rule, signal = str(tags.get(RATED_RULE, "")), str(tags.get(RATED_BY, ""))
            if rule and signal:
                found.setdefault(rule, {})[signal] = float(belief.value)  # type: ignore[arg-type]
        return found

    def vouchers(self, knowledge_base: KnowledgeBase, context_id: str, heuristic: str) -> Mapping[str, float]:
        """Who backed how much of it, or nothing at all where nobody did.

        Nothing is not the same as everybody equally: a heuristic nobody vouched for was written by a finder
        with no economy, and paying its signals out of a share nobody claimed would invent a creditor."""
        found: dict[str, float] = {}
        for belief in knowledge_base.beliefs(context_id, tags=((VOUCHED_FOR, heuristic),)):
            if not isinstance(belief.value, int | float):
                continue
            signal = belief.variable.removesuffix(f" vouched for {heuristic}")
            found[signal] = float(belief.value)  # type: ignore[arg-type]
        return found

    def _keep(self, knowledge_base: KnowledgeBase, context_id: str, signal: str, budget: float, why: str) -> None:
        """The budget as it now stands, with how many times it has moved each way."""
        variable = BUDGET.format(signal=signal)
        belief = knowledge_base.belief(variable, context_id)
        tags = dict(belief.tags) if belief is not None else {}
        tags[why] = int(tags.get(why, 0)) + 1  # type: ignore[arg-type]
        knowledge_base.believe(Belief(variable, context_id, max(0.0, budget), tags=tuple(sorted(tags.items()))))
