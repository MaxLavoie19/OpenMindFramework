import logging
import math
from collections.abc import Mapping, Sequence

from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.constant.model_constant import CURIOSITY

logger = logging.getLogger(__name__)

#: What a model has shown across every judging it has been through, and how many decisions that rests on.
STANDING = "what {model} has shown"
DECISIONS = "decisions"

#: That a model is no longer worth asking.
RETIRED = "{model} is retired"


class ModelRetirement:
    """Which heuristics have shown they are not worth asking any more.

    **The evidence was already being computed and thrown away.** Every judging asks every candidate what it
    would have done, and gets back what it expected of what happened against what knowing nothing would have
    expected. Nothing read that number to remove anything, so the pool grew by a heuristic per ponder per
    price for ever and the judging cost grew with it.

    **Retired and not deleted, because the store is append-only and that is not an accident.** A model is a
    record and what is thought of it is a belief about it; retiring is believing it is not worth asking. So
    what was learned about it survives, a retired model can be un-retired if later evidence says so, and
    nothing has to rewrite a store that is meant to be a record of what was believed when.

    **Read over everything it has been judged on, not over the last round.** One judging is a handful of
    games, and a heuristic can sit below ignorance in one of them by luck. What accumulates is what it has
    shown in total, so a retirement rests on all the evidence there has ever been about it.

    **How much evidence is enough is the caller's.** The settled rule is that coverage governs how long before
    a thing may be dropped and never what it is worth: a heuristic that has answered three decisions and one
    that has answered three hundred can have shown the same, and only the second is known to have. So the
    count gates the retirement and never the number it is judged on."""

    def shown(
        self, knowledge_base: KnowledgeBase, context_id: str, model: str, worth: float, decisions: int
    ) -> tuple[float, int]:
        """What that model has shown in total, once this judging is added to it.

        The worth is the one currency everything here is judged in: what it expected of what was played, less
        what a heuristic with no opinion would have expected. Above nought it saw something; at nought it
        knows nothing; below it, it is actively wrong about what wins."""
        variable = STANDING.format(model=model)
        belief = knowledge_base.belief(variable, context_id)
        held = float(belief.value) if belief is not None and isinstance(belief.value, int | float) else 0.0  # type: ignore[arg-type]
        seen = int(dict(belief.tags).get(DECISIONS, 0)) if belief is not None else 0  # type: ignore[arg-type]
        held, seen = held + worth, seen + decisions
        knowledge_base.believe(Belief(variable, context_id, held, tags=((DECISIONS, seen),)))
        return held, seen

    def retired(self, knowledge_base: KnowledgeBase, context_id: str, model: str) -> bool:
        """Whether that model has been retired."""
        belief = knowledge_base.belief(RETIRED.format(model=model), context_id)
        return bool(belief is not None and belief.value)

    def retire(self, knowledge_base: KnowledgeBase, context_id: str, model: str, why: str) -> None:
        """Believes that model is not worth asking any more, with what it had shown when that was decided."""
        knowledge_base.believe(Belief(RETIRED.format(model=model), context_id, True, tags=(("why", why),)))
        logger.info("Retired %s: %s", model, why)

    def standing(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        worths: Mapping[str, tuple[float, int]],
        least_decided: int,
    ) -> tuple[str, ...]:
        """What each of those models has shown, taken in, and which of them that retires.

        `worths` is one judging's result per model: what it was worth and over how many decisions. A model
        that has answered fewer than `least_decided` decisions in total is left alone however badly it has
        done — not a reprieve, but the difference between a heuristic that has been shown to be useless and
        one that has not been asked enough to say.

        **Nought is the line and it was not chosen here.** A heuristic at nought expected exactly what a
        heuristic with no opinion would have expected, which is to say it knows nothing; below it, it is
        actively wrong about what wins. Both are answers a pool is better off without, and neither needs a
        threshold picked by anybody."""
        gone = []
        for model, (worth, decisions) in worths.items():
            if self.retired(knowledge_base, context_id, model):
                continue
            held, seen = self.shown(knowledge_base, context_id, model, worth, decisions)
            if seen < least_decided or held > 0.0:
                continue
            self.retire(
                knowledge_base,
                context_id,
                model,
                f"it has shown {held:+.3f} over {seen} decisions, and knowing nothing shows nought",
            )
            gone.append(model)
        return tuple(gone)

    def worth_asking_again(self, knowledge_base: KnowledgeBase, context_id: str, model: str, among: int) -> float:
        """The most that model could still turn out to be worth, given how little it has been asked.

        What it has shown per decision, plus how far that could be out — the same bound `ModelDrawer` draws
        with, read over decisions rather than games. Infinite where it has shown nothing yet, which is what no
        evidence means on this scale."""
        belief = knowledge_base.belief(STANDING.format(model=model), context_id)
        if belief is None:
            return math.inf
        seen = int(dict(belief.tags).get(DECISIONS, 0))  # type: ignore[arg-type]
        if seen < 1:
            return math.inf
        held = float(belief.value) if isinstance(belief.value, int | float) else 0.0  # type: ignore[arg-type]
        return held / seen + CURIOSITY * math.sqrt(math.log(max(among, 2)) / seen)

    def keeping(
        self, knowledge_base: KnowledgeBase, context_id: str, records: Sequence[ModelRecord]
    ) -> tuple[ModelRecord, ...]:
        """Those of the records worth asking: never retired, or retired and still possibly worth something.

        **Retirement is a bound and not a door.** A heuristic retired on what it showed over a handful of
        decisions was retired on almost nothing, and the reason it showed nothing may be that the pool it came
        from had no variety for it to be different from. So what keeps it out is not the retirement but the
        arithmetic: what it has shown per decision, plus how far that could still be out, and it comes back
        the moment that is the best thing left to try. One that has been wrong over thousands of decisions has
        a narrow bound and stays out.

        **Nought is the line here too, and it is the same nought.** A heuristic is retired for showing nothing
        where knowing nothing shows nought; it is asked again when it could still show more than nought. No
        second threshold is chosen anywhere.

        A filter and not a registry method, so that nothing which merely lists a context's models starts
        hiding things from whoever asks."""
        kept, back = [], []
        for one in records:
            if not self.retired(knowledge_base, context_id, one.name):
                kept.append(one)
                continue
            if self.worth_asking_again(knowledge_base, context_id, one.name, len(records)) > 0.0:
                kept.append(one)
                back.append(one.name)
        if back:
            logger.info(
                "%d retired heuristics are worth asking again, because what they showed rests on too little "
                "to be sure of: %s",
                len(back), ", ".join(back[:5]) + (" and more" if len(back) > 5 else ""),
            )
        return tuple(kept)
