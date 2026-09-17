from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.training.builder.continuous_trainer_builder import ContinuousTrainerBuilder
from openmind.training.builder.distiller_builder import DistillerBuilder
from openmind.training.builder.rule_selector_builder import RuleSelectorBuilder
from openmind.training.builder.value_distiller_builder import ValueDistillerBuilder
from openmind.training.builder.value_training_loop_builder import ValueTrainingLoopBuilder
from openmind.training.service.continuous_trainer import ContinuousTrainer
from openmind.training.service.distiller import Distiller
from openmind.training.service.rule_selector import RuleSelector
from openmind.training.service.value_distiller import ValueDistiller
from openmind.training.service.value_training_loop import ValueTrainingLoop


def create_distiller(workers: int = 1) -> Distiller:
    """A distiller with its self-play and rule generator, running games and condition checks in that many worker
    processes, and the rater services it measures rules with."""
    return DistillerBuilder().with_workers(workers).build()


def create_rule_selector(workers: int = 1) -> RuleSelector:
    """A rule selector searching in that many worker processes."""
    return RuleSelectorBuilder().with_workers(workers).build()


def create_value_distiller(
    workers: int = 1, memory_cap: MemoryCap | None = None, game_memory: GameMemory | None = None
) -> ValueDistiller:
    """A value distiller with its self-play and value generator, running games and term evaluations in that many worker
    processes, each under the memory cap when given, the valuer services it measures value rules with, and remembering
    every game in the game memory when given."""
    return ValueDistillerBuilder().with_workers(workers).with_memory_cap(memory_cap).with_game_memory(game_memory).build()


def create_value_training_loop(
    workers: int = 1, memory_cap: MemoryCap | None = None, game_memory: GameMemory | None = None
) -> ValueTrainingLoop:
    """A value training loop with its value distiller and match runner, running self-play, term evaluations and games in
    that many worker processes, each under the memory cap when given, and remembering every game in the game memory when
    given."""
    return ValueTrainingLoopBuilder().with_workers(workers).with_memory_cap(memory_cap).with_game_memory(game_memory).build()


def create_continuous_trainer(
    knowledge_base: KnowledgeBase, workers: int = 1, memory_cap: MemoryCap | None = None
) -> ContinuousTrainer:
    """A continuous trainer playing and studying games in that many worker processes, each under the memory cap when
    given, remembering every game in the knowledge base."""
    return ContinuousTrainerBuilder().with_workers(workers).with_memory_cap(memory_cap).with_knowledge_base(knowledge_base).build()
