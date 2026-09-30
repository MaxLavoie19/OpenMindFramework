import logging
import math
import statistics
from collections.abc import Callable, Mapping, Sequence

from openmind.heuristic.model.node import Node
from openmind.training.model.teller_agreement import TellerAgreement

logger = logging.getLogger(__name__)

#: What a heuristic makes of a position, for whoever is being judged. None is knowing nothing about it, which
#: is set aside rather than counted against — the same distinction `AgreementScorer` makes for a decision it
#: cannot answer.
type Valuing = Callable[[Node, str], float | None]


class TellerScorer:
    """Every heuristic put beside a teller over the same positions.

    **It never learns what the teller is.** What arrives is a number per position and nothing about where the
    number came from — a stronger engine, a table, a person. So the same service measures agreement with
    anything that can be asked about a position, and nothing here knows the domain has engines in it.

    **A teller is not an oracle and this is not a grade.** What comes back is how closely the two move
    together, which narrows a field far faster than waiting for games to finish; what it cannot do is pick the
    winner among what survives, because past that point it selects for resembling the teller. The payoff
    decides; this says who is not worth playing.

    **Rank agreement and not raw correlation.** Two evaluations of a position are on different scales and
    nothing makes them comparable — one counts material, another is a network's output in pawns — so what can
    honestly be asked is whether they order the positions the same way. It is also what the measure is *for*:
    a heuristic is used to prefer one position to another, never to name a number.

    It keeps nothing: built once, it is given the positions and the heuristics on every call."""

    def scored(
        self,
        positions: Sequence[tuple[Node, str]],
        told: Sequence[float],
        valuers: Mapping[str, Valuing],
    ) -> tuple[TellerAgreement, ...]:
        """Each named heuristic measured over those positions, against what the teller made of them.

        `positions` is each position with the player it is being judged for, and `told` is the teller's number
        for each in the same order — a position and what somebody who knows says of it. A heuristic that
        values nothing, or values everything alike, has no ordering to compare and is set aside rather than
        scored at nought: not firing is not the same as being wrong."""
        if len(positions) != len(told):
            raise ValueError(f"{len(positions)} positions against {len(told)} tellings")
        found = []
        for name, valuing in valuers.items():
            values = [self._valued(valuing, node, player) for node, player in positions]
            answered = [(one, two) for one, two in zip(values, told, strict=True) if one is not None]
            declined = len(values) - len(answered)
            if len(answered) < 2 or len({one for one, _ in answered}) < 2:
                # It answered too little to be ordered, or gave one number to everything. Both are an opinion
                # that separates nothing, which is what `undecided` means everywhere else here.
                found.append(
                    TellerAgreement(name, 0.0, 0, declined, len(answered), len(positions))
                )
                continue
            found.append(
                TellerAgreement(
                    name,
                    self._together([one for one, _ in answered], [two for _, two in answered]),
                    len(answered),
                    declined,
                    0,
                    len(positions),
                )
            )
        for one in sorted(found, key=lambda held: -held.agreed):
            logger.info(
                "%-44s tracks the teller at %+.3f over %d of %d positions, %d unanswered",
                one.holder[:44], one.agreed, one.decided, one.told, one.declined,
            )
        return tuple(found)

    def _valued(self, valuing: Valuing, node: Node, player: str) -> float | None:
        """What that heuristic makes of that position, or nothing where it cannot say.

        A value that is not a number is knowing nothing — the same answer as None, arrived at by a different
        route, and a scorer that let a NaN through would correlate against it and get a NaN back for the
        whole heuristic."""
        found = valuing(node, player)
        return None if found is None or not math.isfinite(float(found)) else float(found)

    def _together(self, ours: Sequence[float], theirs: Sequence[float]) -> float:
        """How closely the two order the same positions, from -1 to 1.

        Spearman: the values are replaced by their ranks and correlated, so what is measured is the ordering
        and never the units. Ties share the rank they span, which is what keeps a heuristic that says the same
        of several positions from being read as having put them in some order.

        Nought where either side ranks everything alike, which is no ordering to agree with rather than perfect
        disagreement."""
        one, two = self._ranked(ours), self._ranked(theirs)
        if statistics.pstdev(one) == 0.0 or statistics.pstdev(two) == 0.0:
            return 0.0
        return float(statistics.correlation(one, two))

    def _ranked(self, values: Sequence[float]) -> list[float]:
        """Those values as ranks, ties sharing the average of the places they span."""
        order = sorted(range(len(values)), key=lambda at: values[at])
        found = [0.0] * len(values)
        at = 0
        while at < len(order):
            held = at
            while held + 1 < len(order) and values[order[held + 1]] == values[order[at]]:
                held += 1
            shared = (at + held) / 2 + 1
            for one in order[at : held + 1]:
                found[one] = shared
            at = held + 1
        return found
