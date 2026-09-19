from openmind.agent.service.game_memory import GameMemory
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.training.builder.continuous_trainer_builder import ContinuousTrainerBuilder
from openmind.training.builder.value_distiller_builder import ValueDistillerBuilder
from openmind.training.service.continuous_trainer import ContinuousTrainer
from openmind.training.service.value_distiller import ValueDistiller


def create_value_distiller(
    knowledge_base: KnowledgeBase,
    workers: int = 1,
    memory_cap: MemoryCap | None = None,
    game_memory: GameMemory | None = None,
) -> ValueDistiller:
    """A value distiller with its self-play and value generator, running games and term evaluations in that many worker
    processes, each under the memory cap when given, declaring the position rules it fits into the knowledge base, and
    remembering every game in the game memory when given."""
    return (
        ValueDistillerBuilder()
        .with_workers(workers)
        .with_memory_cap(memory_cap)
        .with_game_memory(game_memory)
        .with_knowledge_base(knowledge_base)
        .build()
    )


def create_continuous_trainer(
    knowledge_base: KnowledgeBase, workers: int = 1, memory_cap: MemoryCap | None = None
) -> ContinuousTrainer:
    """A continuous trainer playing and studying games in that many worker processes, each under the memory cap when
    given, remembering every game in the knowledge base."""
    return ContinuousTrainerBuilder().with_workers(workers).with_memory_cap(memory_cap).with_knowledge_base(knowledge_base).build()
