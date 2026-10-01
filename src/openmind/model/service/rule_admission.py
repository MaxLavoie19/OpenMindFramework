import logging
from collections.abc import Mapping, Sequence

from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.model.rule_candidate import RuleCandidate
from openmind.model.model.rule_signal import RuleSignal
from openmind.model.service.rule_budget import RuleBudget
from openmind.model.service.rule_price import RulePrice

logger = logging.getLogger(__name__)


class RuleAdmission:
    """Which candidate rules got vouched for, and by whom.

    **Every signal is paid its income, then spends it on what it wants most.** A signal walks its own
    candidates in the order it rated them and buys while it can afford to, so what it cannot pay for is what it
    wanted least. It does not stop at the first thing it cannot afford: a long rule it has not saved for should
    not cost it the short ones it wanted next.

    **One voucher is enough.** A rule needs somebody willing to stake a budget on it, not a majority — which is
    the whole reason the weights were rejected. So a signal that is right about a narrow class of rules can
    admit them alone, and a rule three signals wanted is admitted once and credited to all three.

    **What is bought is admission, not tenure.** A rule a signal paid for enters the ruleset and then stands on
    its own record: it plays, it is judged with the set, and it can be dropped for performing poorly or for the
    set doing better without it. This says only that somebody was willing to back it."""

    def __init__(self, budget: RuleBudget, price: RulePrice) -> None:
        #: What each signal made of each candidate in the last admission, and what each candidate is called.
        #:
        #: **Held only between `admitted` and `rated`, which is one call apart.** A service here keeps nothing
        #: across callers, and this does not: it is the working of one admission, read once by whoever asked
        #: for it. Written as state rather than returned because `admitted` already answers a different
        #: question and two answers in one return is how a caller comes to ignore the second.
        self._rated: dict[str, tuple[float, ...]] = {}
        self._candidates: list[str] = []
        self._budget = budget
        self._price = price

    @property
    def budget(self) -> RuleBudget:
        """The ledger it spends from, which is also where a heuristic's backers are recorded for paying
        later. One ledger and not two: a caller settling a debt has to settle it in the same purse."""
        return self._budget

    def admitted(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        candidates: Sequence[RuleCandidate],
        signals: Sequence[RuleSignal],
        vocabulary: int,
    ) -> Mapping[int, tuple[str, ...]]:
        """The candidates that were bought, by their place in `candidates`, each with who bought it.

        `vocabulary` is how many readings the search drew from, which is what a rule's price is measured
        against. Nothing is admitted where there are no signals — an economy with nobody in it buys nothing,
        which is a truthful answer rather than a reason to let everything through."""
        if not candidates or not signals:
            return {}
        self._budget.allowed(knowledge_base, context_id, [one.name for one in signals])
        prices = [self._price.priced(one.expression, vocabulary) for one in candidates]
        bought: dict[int, list[str]] = {}
        self._rated = {}
        self._candidates = [str(getattr(one.rule, "source", "") or "") for one in candidates]
        for signal in signals:
            rates = signal.rates(candidates)
            # Kept as they are computed, because they are the only account of *why* a rule is in a heuristic
            # and they were being thrown away the moment they had been spent on. Every signal's opinion and
            # not only the buyers': a signal rating a rule at nearly nothing says as much about that rule as
            # one that paid for it.
            self._rated[signal.name] = tuple(float(one) for one in rates)
            wanted = sorted(range(len(candidates)), key=lambda at: -rates[at])
            spent, took = 0.0, 0
            for at in wanted:
                if rates[at] <= 0.0:
                    continue
                if not self._budget.vouched(knowledge_base, context_id, signal.name, prices[at]):
                    continue
                bought.setdefault(at, []).append(signal.name)
                spent += prices[at]
                took += 1
            logger.info(
                "%s bought %d of %d candidates for %.1f bits, with %.1f left",
                signal.name,
                took,
                len(candidates),
                spent,
                self._budget.held(knowledge_base, context_id, signal.name),
            )
        logger.info(
            "%d of %d candidates were vouched for, %d of them by more than one signal",
            len(bought),
            len(candidates),
            sum(1 for who in bought.values() if len(who) > 1),
        )
        return {at: tuple(who) for at, who in bought.items()}

    def rated(self, admitted: Mapping[int, tuple[str, ...]]) -> Mapping[str, Mapping[str, float]]:
        """What every signal made of each rule that was bought, by the rule's own name.

        **The latest and never a total.** A ponder rates afresh over its own rows, so what a signal thought
        last time is not evidence about this fit — it is a different fit. Nothing accumulates here.

        Named by what the rule reads, because that is what it is called everywhere else and what a page will
        show. Empty before anything has been admitted, which is the honest answer rather than a table of
        noughts."""
        found: dict[str, dict[str, float]] = {}
        for at in admitted:
            name = self._named(at)
            if not name:
                continue
            found[name] = {
                signal: rates[at] for signal, rates in self._rated.items() if at < len(rates)
            }
        return found

    def _named(self, at: int) -> str:
        """What the candidate in that place is called, which is what it reads."""
        return self._candidates[at] if at < len(self._candidates) else ""

    def shares(self, admitted: Mapping[int, tuple[str, ...]]) -> Mapping[str, float]:
        """How much of what was admitted each signal vouched for, which is what `RuleBudget.earned` pays on.

        **A rule two signals bought is half each.** What a heuristic turned out to be worth is a fact about the
        whole of it, so what separates the signals is how much of it each one backed — and a signal that was
        one of two backers backed half of that rule, not all of it. Shares sum to one where anything was
        bought, so the whole worth is paid out and no more."""
        earned: dict[str, float] = {}
        for who in admitted.values():
            for name in who:
                earned[name] = earned.get(name, 0.0) + 1.0 / len(who)
        whole = sum(earned.values())
        return {} if whole <= 0 else {name: held / whole for name, held in earned.items()}
