from pathlib import Path

from openmind.dashboard.model.heuristic_standing import HeuristicStanding
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.model.service.model_retirement import RETIRED
from openmind.model.service.rule_budget import VOUCHED_FOR
from openmind.training.service.judging_record import (
    DECIDED, DECLINED, JUDGED, JUDGINGS, MASS, OFFERED, TOLD, TOLD_OF, UNDECIDED,
)

#: What the stored names look like, so a heuristic can be found from either end.
#:
#: Beliefs are kept under a sentence with the heuristic's name in it, which is how they read in a store
#: somebody opens by hand. Reading them back means taking the name out again, and that is done by the two
#: fixed ends rather than by splitting on anything inside — a heuristic is called `position value at price
#: 0.001, fit 9`, and there is nothing in that a split could safely rely on.
JUDGED_BEFORE, JUDGED_AFTER = JUDGED.split("{model}")
TOLD_BEFORE, TOLD_AFTER = TOLD.split("{model}")
RETIRED_BEFORE, RETIRED_AFTER = RETIRED.split("{model}")


class HeuristicStandingReader:
    """Every heuristic a run has judged, with what each measure made of it.

    **Read from the knowledge base and not from the log**, because what a page shows has to outlive the run
    that produced it: a restart loses nothing, and two runs against one store are seen together.

    It keeps nothing: built once, it is given a directory on every call."""

    def standings(self, directory: Path, domain: str) -> tuple[HeuristicStanding, ...]:
        """Every heuristic judged at least once, the ones worth most first.

        Sorted by what they have been worth rather than by name, because a page of forty candidates is read to
        find which are working. A heuristic the teller has an opinion about but the games have not yet judged
        still appears, at nought — not having been measured is a different thing from having measured badly."""
        if not (directory / domain).is_dir():
            return ()
        knowledge_base = create_knowledge_base(domain, directory)
        context = knowledge_base.context_named(domain)
        if context is None:
            return ()
        found: dict[str, dict] = {}
        for belief in knowledge_base.beliefs(context.id):
            name = self._named(str(belief.variable), JUDGED_BEFORE, JUDGED_AFTER)
            if name is not None:
                found.setdefault(name, {}).update(self._judged(belief))
                continue
            name = self._named(str(belief.variable), TOLD_BEFORE, TOLD_AFTER)
            if name is not None:
                found.setdefault(name, {}).update(self._told(belief))
                continue
            name = self._named(str(belief.variable), RETIRED_BEFORE, RETIRED_AFTER)
            if name is not None:
                found.setdefault(name, {})["retired"] = bool(belief.value)
                continue
            vouched = dict(belief.tags).get(VOUCHED_FOR)
            if vouched is not None:
                held = found.setdefault(str(vouched), {})
                held["vouched"] = (*held.get("vouched", ()), str(belief.variable).split(" vouched for ")[0])
        return tuple(
            sorted(
                (HeuristicStanding(name=name, **held) for name, held in found.items()),
                key=lambda one: (-one.worth, one.name),
            )
        )

    def _named(self, variable: str, before: str, after: str) -> str | None:
        """The heuristic a belief is about, or None where the belief is about something else."""
        if not variable.startswith(before) or not variable.endswith(after):
            return None
        held = variable[len(before): len(variable) - len(after) if after else None]
        return held or None

    def _judged(self, belief) -> dict:
        """What a judging belief says, with the worth worked out from its two halves.

        The worth is not stored because it is not a fact of its own: it is the mass less what a heuristic with
        no opinion would have expected, and keeping it beside the two would be one number that can disagree
        with them."""
        tags = dict(belief.tags)
        mass, offered = self._number(tags.get(MASS)), self._number(tags.get(OFFERED))
        return {
            "mass": mass,
            "offered": offered,
            "worth": mass - offered,
            "decided": int(self._number(tags.get(DECIDED))),
            "declined": int(self._number(tags.get(DECLINED))),
            "undecided": int(self._number(tags.get(UNDECIDED))),
            "judgings": int(self._number(tags.get(JUDGINGS))),
        }

    def _told(self, belief) -> dict:
        tags = dict(belief.tags)
        return {"tracks": self._number(belief.value), "told": int(self._number(tags.get(TOLD_OF)))}

    def _number(self, held: object) -> float:
        return float(held) if isinstance(held, int | float) and not isinstance(held, bool) else 0.0
