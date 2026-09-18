from typing import Self

from openmind.agent.service.game_memory import GameMemory
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.value_distiller import ValueDistiller
from openmind.world.service.state_reader import StateReader


class ValueDistillerBuilder:
    """Sets how many worker processes a value distiller's self-play games and term evaluations run in, 1 by
    default, and the memory each of them holds at most, no cap by default, and wires the services it works with."""

    def __init__(self) -> None:
        self._workers = 1
        self._memory_cap: MemoryCap | None = None
        self._game_memory: GameMemory | None = None
        self._knowledge_base: KnowledgeBase | None = None

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def with_memory_cap(self, memory_cap: MemoryCap | None) -> Self:
        self._memory_cap = memory_cap
        return self

    def with_game_memory(self, game_memory: GameMemory | None) -> Self:
        """Where every game is remembered as it ends; None, the default, remembers none."""
        self._game_memory = game_memory
        return self

    def with_knowledge_base(self, knowledge_base: KnowledgeBase) -> Self:
        """Where the fitted position rules are declared, and where they are read back from."""
        self._knowledge_base = knowledge_base
        return self

    def build(self) -> ValueDistiller:
        if self._knowledge_base is None:
            raise ValueError("A value distiller needs a knowledge base to declare its rules into")
        state_reader = StateReader()
        return ValueDistiller(
            SelfPlay(state_reader, TaskRunner(self._workers, self._memory_cap)),
            ValueGeneratorBuilder().with_workers(self._workers).with_memory_cap(self._memory_cap).build(),
            PositionRowMapper(state_reader),
            self._knowledge_base,
            self._game_memory,
        )
