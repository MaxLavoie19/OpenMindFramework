import json
import logging
from dataclasses import replace

from openmind.agent.model.domain import Domain
from openmind.doxastic.constant.doxastic_constant import PROVED
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.inference.model.deduction import Deduction
from openmind.inference.model.expression import Expression
from openmind.rbs.model.value_base import ValueBase
from openmind.training.constant.continuous_constant import PROOF_KEYWORD, SEED_KEYWORD
from openmind.training.constant.signal_constant import WEIGHTED, WIN
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.game_lesson import GameLesson
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.service.online_value_fitter import OnlineValueFitter
from openmind.training.service.signal_recorder import SignalRecorder

logger = logging.getLogger(__name__)


class LessonLearner:
    """Learns from a game's lesson in the process that runs the training, as the lesson arrives:

    1. every candidate's readings add to its record: agreements, disagreements, reliability;
    2. the lesson's proofs and seeds join the pool the rule search starts from, and each proof is remembered in the
       knowledge base as `proved`, with the game and the ply of its position;
    3. every arm's value base moves one step toward the targets of the signal it follows, an arm not named after a
       followed signal, such as a deduced one, toward the weighted aggregation's, or winning's without it.

    The pool keeps every proof and seed of the run, the most recent seeds first."""

    def __init__(
        self,
        signal_recorder: SignalRecorder,
        online_value_fitter: OnlineValueFitter,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        self._signal_recorder = signal_recorder
        self._online_value_fitter = online_value_fitter
        self._knowledge_base = knowledge_base
        self._deductions: list[Deduction] = []
        self._seeds: dict[str, Expression] = {}

    @property
    def deductions(self) -> tuple[Deduction, ...]:
        return tuple(self._deductions)

    @property
    def seeds(self) -> tuple[Expression, ...]:
        return tuple(reversed(self._seeds.values()))

    def learn(
        self, domain: Domain, library: SignalLibrary, lesson: GameLesson, label: str, settings: ContinuousTrainingSettings
    ) -> SignalLibrary:
        library = self._signal_recorder.add(library, lesson.candidates, lesson.readings)
        proven = self._pool(domain, lesson, label)
        price = sorted(settings.values.prices, reverse=True)[len(settings.values.prices) // 2]
        bases: list[tuple[str, ValueBase]] = []
        largest, moved = 0.0, None
        for name, base in library.value_bases:
            targets = lesson.targets.get(name, lesson.targets.get(WEIGHTED, lesson.targets.get(WIN)))
            stepped = None if targets is None else self._online_value_fitter.step(base, lesson.terms, targets, settings.learning_rate, price)
            if stepped is None:
                bases.append((name, base))
                continue
            bases.append((name, stepped[0]))
            if stepped[1] >= largest:
                largest, moved = stepped[1], name
        logger.info(
            "Learned from %s: %d anchors read by %d signals, %d positions proven, %d seeds; %s",
            label,
            lesson.readings.anchors,
            len(lesson.candidates),
            proven,
            0 if lesson.pondering is None else len(lesson.pondering.seeds),
            "no weight step: no arm's rules could all be read on the game, as when a search replaced them while it was played"
            if moved is None
            else f"the largest weight step {largest:.6g} on {moved}",
        )
        return replace(library, value_bases=tuple(bases))

    def _pool(self, domain: Domain, lesson: GameLesson, label: str) -> int:
        """Adds the lesson's proofs and seeds to the pool, remembering each proof; how many positions were proven."""
        pondering = lesson.pondering
        if pondering is None:
            return 0
        proofs = [
            deduction
            for deduction in (*pondering.deductions, *(deduction for walk in pondering.walks for deduction in walk.deductions))
            if deduction.payoffs is not None
        ]
        plies = {state: ply for ply, state in enumerate(lesson.game.states)}
        for deduction in proofs:
            self._deductions.append(deduction)
            if self._knowledge_base is not None:
                ply = plies.get(deduction.state)
                self._knowledge_base.remember(
                    Record(
                        json.dumps({"game": label, "ply": ply, "player": deduction.player, "payoffs": list(deduction.payoffs)}),  # type: ignore[arg-type]
                        Provenance(PROVED, game=label, ply=ply),
                        (domain.name,),
                        keywords=(PROOF_KEYWORD,),
                    )
                )
        for seed, source in zip(pondering.seeds, pondering.sources, strict=True):
            self._seeds.pop(source.source, None)
            self._seeds[source.source] = seed
            if self._knowledge_base is not None:
                self._knowledge_base.remember(
                    Record(source.source, Provenance(PROVED, game=label), (domain.name,), keywords=(SEED_KEYWORD,))
                )
        return len(proofs)
