import logging
import math
import random
from collections.abc import Sequence

from openmind.inference.constant.certainty_constant import ACCURACY, SCORED
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.knowledge.model.model_record import ModelRecord
from openmind.model.service.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

#: How far a model that has hardly played is preferred over one that has done well. The usual constant of the
#: bound, kept rather than chosen: it is the one that makes the regret of the whole run grow like the log of
#: its length, which is the property the bound is for.
CURIOSITY = math.sqrt(2.0)


class ModelDrawer:
    """Which model plays next: drawn, not picked.

    **`best` answers a different question and would answer it forever.** It gives the model measured most
    accurate, which is what to play when the game matters — a tournament, an answer somebody acts on. Asked
    every game of a training run it would play the first thing that ever won and never try anything else, so a
    heuristic that lost its opening three games could never be shown to be good and a new one could never be
    shown at all.

    **A model never played has everything to prove, so it goes first.** Its bound is infinite, which is not a
    trick: it is what "no evidence" means on this scale, and it is why a pool that grows keeps being explored
    rather than settling on whatever happened to be registered early.

    **Drawn rather than taken, because two sides need two.** The bound says what is worth trying; the draw
    turns that into a choice, and a choice that is only ever the maximum makes both sides of a game the same
    model, which settles nothing about either. Weighted by the bound, so what is worth trying is tried often
    and the rest is tried sometimes."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry

    def drawn(
        self,
        knowledge_base: KnowledgeBase,
        context_id: str,
        task: str,
        rng: random.Random,
        how_many: int = 1,
        registry: ModelRegistry | None = None,
    ) -> tuple[ModelRecord, ...]:
        """That many models of the task, each drawn by its bound, without drawing one twice.

        Fewer come back than were asked for where the task has fewer models, and none at all where it has
        none — a caller wanting two and given one is being told something true about the pool."""
        held = registry or self._registry
        if held is None:
            raise ValueError("A drawer needs a registry, given here or when it was built")
        found = held.of_task(knowledge_base, context_id, task)
        if not found:
            return ()
        bounds = [self.bound(knowledge_base, one, len(found), held) for one in found]
        drawn = tuple(found[at] for at in self.among(bounds, rng, how_many))
        logger.debug(
            "Drew %s to play %s",
            ", ".join(knowledge_base.readable_model(one.id) for one in drawn),
            task,
        )
        return drawn

    def among(self, bounds: Sequence[float], rng: random.Random, how_many: int = 1) -> tuple[int, ...]:
        """Which of those to try, drawn by their bounds, without drawing one twice.

        **The draw said over bounds alone, because the bounds are not always to hand where the draw is.** A
        caller playing games in other processes cannot ask the knowledge base as each game starts — the store
        is being written to meanwhile, and a reader racing a writer is a bug waiting for a busy night. It
        reads what each candidate is worth trying when it is safe to, and draws from that reading as often as
        it likes. `drawn` is this over bounds it works out itself.

        Fewer come back than were asked for where there are fewer to draw from, which is something true about
        the pool rather than an error to raise about."""
        held = list(bounds)
        places = list(range(len(held)))
        drawn: list[int] = []
        for _ in range(min(how_many, len(held))):
            at = self._roulette(held, rng)
            drawn.append(places.pop(at))
            held.pop(at)
        return tuple(drawn)

    def bound(
        self,
        knowledge_base: KnowledgeBase,
        model: ModelRecord,
        among: int,
        registry: ModelRegistry | None = None,
    ) -> float:
        """What that model is worth trying: what it has done, plus what is not yet known about it.

        Infinite where it has never been scored, which is the whole of the second term when nothing has been
        counted. `among` stands in for how much has been played in total — the bound wants the log of that,
        and a pool's size is what a caller has to hand without counting every game again."""
        held = registry or self._registry
        measure = held.measured(knowledge_base, model) if held is not None else None
        scored = self._scored(knowledge_base, model)
        if not scored:
            return math.inf
        done = 0.0 if measure is None or measure.accuracy is None else measure.accuracy
        return done + CURIOSITY * math.sqrt(math.log(max(among, 2)) / scored)

    def _scored(self, knowledge_base: KnowledgeBase, model: ModelRecord) -> int:
        """How many games that model has been scored over, nought where none.

        Read off the belief its accuracy is kept on, which is where the count already lives — the measure
        carries the accuracy and how many readings were timed, and neither of those is how often it played."""
        for subject in (model.id, model.mechanism):
            belief = knowledge_base.belief(ACCURACY.format(mechanism=subject), model.context)
            if belief is not None:
                held = int(dict(belief.tags).get(SCORED, 0))  # type: ignore[arg-type]
                if held:
                    return held
        return 0

    def _roulette(self, bounds: Sequence[float], rng: random.Random) -> int:
        """One of them, the chance of each being its share of the bounds.

        **Anything unbounded is taken first**, and where several are, one of those at random. A share of
        infinity is not a number, and a model with everything to prove should not have to win a lottery to be
        tried once."""
        unproven = [at for at, one in enumerate(bounds) if math.isinf(one)]
        if unproven:
            return rng.choice(unproven)
        total = sum(bounds)
        if total <= 0:
            return rng.randrange(len(bounds))
        wanted, running = rng.random() * total, 0.0
        for at, one in enumerate(bounds):
            running += one
            if running >= wanted:
                return at
        return len(bounds) - 1
