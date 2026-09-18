from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.training.builder.value_training_loop_builder import ValueTrainingLoopBuilder
from openmind.training.service.value_training_loop import ValueTrainingLoop


def test_build_gives_a_value_training_loop(knowledge: KnowledgeBase) -> None:
    assert isinstance(
        ValueTrainingLoopBuilder().with_workers(2).with_knowledge_base(knowledge).build(), ValueTrainingLoop
    )
