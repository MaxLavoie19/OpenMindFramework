from typing import Self

from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.task_runner import TaskRunner
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.continuous_trainer import ContinuousTrainer
from openmind.training.service.ending_walker import EndingWalker
from openmind.training.service.game_study import GameStudy
from openmind.training.service.self_play import SelfPlay
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.service.state_reader import StateReader


class ContinuousTrainerBuilder:
    """Sets how many worker processes continuous training plays and studies its games in, 1 by default; the memory each of
    them holds at most, no cap by default; and the knowledge base every game and proof is remembered in, which it needs.
    A game's study runs inside its worker, on services of its own."""

    def __init__(self) -> None:
        self._workers = 1
        self._memory_cap: MemoryCap | None = None
        self._knowledge_base: KnowledgeBase | None = None

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def with_memory_cap(self, memory_cap: MemoryCap | None) -> Self:
        self._memory_cap = memory_cap
        return self

    def with_knowledge_base(self, knowledge_base: KnowledgeBase) -> Self:
        self._knowledge_base = knowledge_base
        return self

    def build(self) -> ContinuousTrainer:
        if self._knowledge_base is None:
            raise ValueError("Continuous training needs a knowledge base to remember its games in")
        state_reader = StateReader()
        here = TaskRunner(1)
        study = GameStudy(
            SelfPlay(state_reader, here),
            EndingWalker(PositionDeducer(state_reader, ActionTextMapper())),
        )
        return ContinuousTrainer(
            study,
            GameMemory(self._knowledge_base),
            self._knowledge_base,
            SelfPlay(state_reader, here),
            ArmSelector(),
            TaskRunner(self._workers, self._memory_cap),
        )
