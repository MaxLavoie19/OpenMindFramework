import logging
from collections.abc import Callable, Sequence

from openmind.rbs.factory.rbs_factory import create_rule_heuristic
from openmind.heuristic.service.rule_position_valuer import RulePositionValuer
from openmind.training.model.decided import Decided
from openmind.training.model.teller_agreement import TellerAgreement
from openmind.training.service.teller_scorer import TellerScorer

logger = logging.getLogger(__name__)

#: How a position is named for whoever is being asked about it. The teller is outside OMF and speaks the
#: domain's own language, so this is the one place a caller says how to put a position into it.
type Naming = Callable[[Decided], str | None]

#: What somebody who knows makes of those positions, in the order given, nothing where they would not say.
type Telling = Callable[[Sequence[str]], Sequence[float | None]]


class TellerJudging:
    """Every heuristic put beside a teller over the positions a set of decisions was made in.

    **The dense half of judging.** Asking what a heuristic expected of a game costs candidates times decisions
    times the moves on offer, because every move's position has to be valued to place the one played among
    them. Asking what it makes of the position itself costs candidates times decisions — the moves drop out,
    which in chess is about thirty-five times less work for a measurement that says something on every
    position rather than once per game.

    **It narrows and does not decide.** A heuristic that tracks a teller perfectly has learned the teller,
    blind spots included, and what is wanted is winning. So what this is for is finding who is not worth the
    expensive question, and the payoff settles the rest.

    It keeps nothing: built once, it is given the decisions, the models and the teller on every call."""

    def __init__(self, reading: float = 1.0, seconds: float = 0.0) -> None:
        self._reading = reading
        self._seconds = seconds

    def judged(
        self,
        decisions: Sequence[Decided],
        models: Sequence[tuple[str, object]],
        naming: Naming,
        telling: Telling,
    ) -> tuple[TellerAgreement, ...]:
        """Each named model measured against what the teller made of the same positions.

        Only the positions the teller would speak about are kept, and they are kept together with the
        decisions they came from — a position it declined cannot be a row, and dropping it from one list and
        not the other is how a column comes to be measured against the wrong board."""
        asked = [(one, naming(one)) for one in decisions]
        named = [(one, held) for one, held in asked if held]
        if not named:
            return ()
        said = telling([held for _, held in named])
        kept = [(one, value) for (one, _), value in zip(named, said, strict=True) if value is not None]
        if len(kept) < 2:
            logger.info(
                "The teller said something about %d of %d positions, which is too few to be ordered",
                len(kept), len(decisions),
            )
            return ()
        logger.info(
            "The teller read %d of %d positions; %d were not put to it and %d it would not say",
            len(kept), len(decisions), len(decisions) - len(named), len(named) - len(kept),
        )
        positions = [(one.node, one.player) for one, _ in kept]
        return TellerScorer().scored(
            positions, [value for _, value in kept], {name: self._valuing(model) for name, model in models}
        )

    def _valuing(self, model: object):
        """That model as a way of valuing a position for a player.

        A valuer of its own per model, so each keeps its own reading costs — the same reason the dispatcher
        builds one per worker rather than sharing one."""
        valuer = RulePositionValuer(create_rule_heuristic(self._reading, self._seconds, {}))

        def valuing(node, player: str) -> float | None:
            found = valuer.values(model, node)  # type: ignore[arg-type]
            if found is None:
                return None
            names = node.game.players().names  # type: ignore[union-attr]
            return found[names.index(player)] if player in names else None

        return valuing
