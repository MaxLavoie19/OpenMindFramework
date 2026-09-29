import logging
import math
import statistics
from collections.abc import Callable, Mapping, Sequence

from openmind.heuristic.model.node import Node
from openmind.training.model.agreement import Agreement
from openmind.training.model.decided import Decided
from openmind.world.model.action import Action

logger = logging.getLogger(__name__)

#: How a rating becomes a chance. The scores a heuristic gives are on no particular scale — one rule-based
#: system counts squares and another counts material — so they are made comparable the way a preference over
#: choices already is here, by softmax. A heuristic that rates everything alike comes out uniform, which is
#: what knowing nothing should look like, and one that rates a move far above the rest comes out near certain.
type Rating = Callable[[Node, tuple[Action, ...], str], tuple[float | None, ...]]


class AgreementScorer:
    """What every heuristic would have expected of a game that was played, against what happened in it.

    **A heuristic need not have played to be scored.** Two played the game; the candidates are many, and
    replaying the game asks all of them at once. So how fast a heuristic gathers evidence is not bounded by
    how often it is drawn to play, which is what makes drawing at random affordable: a candidate that never
    gets a turn is still measured every game anybody plays.

    **The same service learns from other players.** Handed a game somebody else played, it asks which
    heuristics would have chosen their moves. Nothing here knows whose game it is — only what was done and
    what it paid — so watching a stronger player and scoring one's own candidates are one mechanism.

    **It asks for a rating and never for a model.** A rule-based system fills the port, and so does a network,
    and so does a table; this sees a function from a position and its actions to a number each, and nothing
    else. That is what lets one number compare model families, which is the only comparison that must.

    **Unfinished games are not handed here.** A game cut off pays nobody, so agreeing with either side is
    evidence of nothing. That is a fact about the game and the caller's to notice rather than a case to
    handle quietly — counted as a draw it would make every abandoned game look like a fought one."""

    def scored(self, decisions: Sequence[Decided], raters: Mapping[str, Rating]) -> tuple[Agreement, ...]:
        """Each named heuristic measured over those decisions, in the order they were given.

        A decision it cannot answer, or answers without separating anything, is set aside rather than counted
        against it. What it expected of the rest is gathered as the chance it gave what was taken, weighted by
        what the game paid whoever took it."""
        found = []
        for holder, rating in raters.items():
            mass, decided, declined, undecided, offered = 0.0, 0, 0, 0, 0.0
            for one in decisions:
                chances = self._chances(rating, one)
                if chances is None:
                    declined += 1
                    continue
                at = one.offered.index(one.taken) if one.taken in one.offered else None
                if at is None:
                    # The move played is not among the actions it was shown. Nothing it said bears on what
                    # happened, so this is no more a mistake of the heuristic's than a decline is.
                    declined += 1
                    continue
                if self._alike(chances):
                    undecided += 1
                    continue
                decided += 1
                mass += chances[at] * one.paid
                offered += one.paid / len(one.offered)
            found.append(Agreement(holder, mass, decided, declined, undecided, offered))
        for one in sorted(found, key=lambda held: -(held.mass - held.offered)):
            logger.info(
                "%s expected %.2f of what happened where knowing nothing would have expected %.2f, over %d "
                "decisions of %d; it had nothing to say in %d and nothing to choose between in %d",
                one.holder, one.mass, one.offered, one.decided,
                len(decisions), one.declined, one.undecided,
            )
        return tuple(found)

    def _chances(self, rating: Rating, decision: Decided) -> tuple[float, ...] | None:
        """What it gives each action on offer, as chances summing to one; None where it rated none of them.

        **Saying nothing about an action is saying it is unremarkable, which is nought.** A rating adds to or
        takes from what an action is worth, so the absence of one is no change — and nought is what no change
        is on that scale. Softmax over all-noughts is uniform, so a heuristic silent throughout comes out
        knowing nothing, which is what it does know.

        **Read any other way it silences the rules that fire once.** Taking the least rating given, which this
        did: a rule that says a move blunders mate at minus a hundred and says nothing of the rest makes every
        action minus a hundred, comes out uniform, and is recorded as having no opinion — the rule whose whole
        content is that one move is ruinous. A rule that rates one move highly and stays quiet goes the same
        way. Both are the shape this is meant to keep."""
        scored = rating(decision.node, decision.offered, decision.player)
        if len(scored) != len(decision.offered) or all(one is None for one in scored):
            return None
        held = self._standardised([0.0 if one is None else one for one in scored])
        # Softmax, shifted by the largest so nothing overflows.
        most = max(held)
        weighed = [math.exp(one - most) for one in held]
        total = sum(weighed)
        return tuple(one / total for one in weighed)

    def _standardised(self, values: Sequence[float]) -> list[float]:
        """Those ratings divided by how much they vary, so that only their shape reaches the softmax.

        **Without this a ruleset scores better by shouting.** Softmax reads a difference between ratings, and
        a ruleset that multiplies all its weights by a hundred multiplies every difference by a hundred, so it
        comes out near-certain wherever it has any opinion at all. Measured over four thousand decisions with
        the skill held fixed, a ruleset scoring 0.120 at its own scale scored 0.323 at a hundred times it —
        2.7 times the mass for no more skill, enough to beat an honest ruleset two and a half times better.

        **And the sweep manufactures exactly that.** Each L1 price is fitted separately and shrinks its
        coefficients differently — one price kept 139 terms unshrunk where another kept 10 heavily shrunk — so
        the least-priced fit would win the judging by construction rather than by merit.

        **The regulariser belongs here and not on the weights.** Shrinking what a rule may weigh would fight
        the checkmate finder, which needs an enormous coefficient so that on the rare position where it fires
        it swamps everything else; that is a rule marked down for being decisive, which is the error this
        project has already struck once. Standardising where the comparison happens removes what shouting buys
        without constraining what any rule may say. Measured after: a hundred times the scale scores
        identically, and skill still separates.

        Left alone where nothing varies, since there is nothing to divide by and `_alike` already reads that
        as a heuristic with no opinion."""
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        return list(values) if spread <= 0.0 or not math.isfinite(spread) else [one / spread for one in values]

    def _alike(self, chances: Sequence[float]) -> bool:
        """Whether it rated everything the same, which is an opinion that separates nothing.

        Counted apart from a decline because the two are different states — one heuristic could not read the
        position and the other read it and found nothing to choose — and counted against neither, because a
        tie is not a wrong answer."""
        return len(chances) < 2 or max(chances) - min(chances) < 1e-12
