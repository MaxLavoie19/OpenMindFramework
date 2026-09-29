import logging
import math
import random
from collections.abc import Mapping, Sequence

from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

#: What a way of making a heuristic has been worth, by factor and value.
WORTH = "worth of {factor} being {value}"

#: The tag its count is kept under, so a mean can be moved without keeping every number behind it.
MADE = "made"


class FactorWeights:
    """What each way of making a heuristic has been worth, so the next one is made the way that has paid.

    **A factor is a choice the making offers**, and its values are the ways that choice can go: the price a
    fit is swept at, the signal that admits a term, the target it is fitted to. Each value starts equal and
    gains weight by the measured quality of the heuristics made with it.

    **The alternative was trying every combination, and that is an experiment rather than a learner.** Three
    prices under five admissions is fifteen heuristics a ponder, each to be judged against every game that
    ended, where what is wanted is one heuristic made the way that has worked. Drawing a value per factor
    makes one, and the evidence for that way of making accumulates over ponders instead of over a grid.

    **The currency is the one everything else here is judged in** — what a heuristic expected of what
    happened, against what knowing nothing would have expected. Nothing new is measured: the judging already
    computes it every position, and this only says where to send it.

    **Drawn rather than taken, for the reason `ModelDrawer` is.** A value never tried has everything to prove
    and goes first; one that has paid is tried often and the rest are tried sometimes — a value that looked
    poor early must be able to come back, or the first accident becomes the policy.

    **The bound is the drawer's; the draw is not.** A model's accuracy is never negative, so the drawer can
    take a share of the total. A way of making can be worth less than nothing, and shares of that are not
    shares — so the choosing is by softmax here, which is what this project already uses to turn ratings on
    no particular scale into chances."""

    def drawn(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        factor: str,
        values: Sequence[str],
        rng: random.Random,
    ) -> str:
        """Which way to make the next heuristic, drawn by what each way has been worth.

        The values are the caller's: a factor knows its own choices and this cannot. Given one it gives that
        one back, which is a caller saying there is no choice here rather than an error."""
        if not values:
            raise ValueError(f"A factor needs values to draw between; {factor} was given none")
        if len(values) == 1:
            return values[0]
        bounds = [self.bound(knowledge_base, context_id, factor, one, len(values)) for one in values]
        at = self._chosen(bounds, rng)
        logger.debug("Making the next heuristic with %s being %s", factor, values[at])
        return values[at]

    def bound(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        factor: str,
        value: str,
        among: int,
    ) -> float:
        """What that way of making is worth trying: what it has been worth, plus what is not known about it.

        Infinite where nothing has been made that way, which is what no evidence means on this scale."""
        belief = knowledge_base.belief(WORTH.format(factor=factor, value=value), context_id)
        made = 0 if belief is None else int(dict(belief.tags).get(MADE, 0))  # type: ignore[arg-type]
        if not made:
            return math.inf
        worth = float(belief.value) if isinstance(belief.value, int | float) else 0.0  # type: ignore[arg-type]
        return worth + math.sqrt(2.0) * math.sqrt(math.log(max(among, 2)) / made)

    def paid(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        made_by: Mapping[str, str],
        worth: float,
    ) -> None:
        """What a heuristic turned out to be worth, credited to every way it was made.

        **Every factor that made it takes the whole of it, and none takes a share.** Splitting the credit
        would need to know how much of the heuristic each choice accounted for, which is the question the
        drawing exists because nobody can answer. What separates the factors instead is that each is drawn
        independently, so over many heuristics a value that pays only alongside one other value is told apart
        from one that pays whatever it is drawn with.

        Kept as a running mean, because what is wanted is what that way of making is worth and not what the
        last one happened to be. A way that has made forty heuristics moves by a fortieth of the difference."""
        for factor, value in made_by.items():
            variable = WORTH.format(factor=factor, value=value)
            belief = knowledge_base.belief(variable, context_id)
            made = 0 if belief is None else int(dict(belief.tags).get(MADE, 0))  # type: ignore[arg-type]
            held = (
                0.0
                if belief is None or not isinstance(belief.value, int | float)
                else float(belief.value)  # type: ignore[arg-type]
            )
            knowledge_base.believe(
                Belief(
                    variable,
                    context_id,
                    held + (worth - held) / (made + 1),
                    tags=((MADE, made + 1),),
                )
            )
        logger.debug("Credited %.3f to %s", worth, ", ".join(f"{one} being {two}" for one, two in made_by.items()))

    def _chosen(self, bounds: Sequence[float], rng: random.Random) -> int:
        """Which to make with, drawn by its bound: anything unproven first, and the rest by softmax.

        **Not a share of the total, which is what the drawer takes over models.** A model's accuracy is never
        negative, so a share of it is a share. A way of making can be worth less than nothing — that is what
        it means for the heuristics it made to have expected the losing move more often than ignorance would
        have — and shifting the worst to nought to make shares of it gives it no share at all, so the first
        unlucky measurement becomes the last word. Written that way first, and the tests said so.

        Softmax is how this project already turns ratings on no particular scale into chances, for the same
        reason `AgreementScorer` does it: nothing is ever impossible, and what is worth more is drawn more."""
        unproven = [at for at, one in enumerate(bounds) if math.isinf(one)]
        if unproven:
            return rng.choice(unproven)
        most = max(bounds)
        weighed = [math.exp(one - most) for one in bounds]
        wanted, running = rng.random() * sum(weighed), 0.0
        for at, one in enumerate(weighed):
            running += one
            if running >= wanted:
                return at
        return len(bounds) - 1
