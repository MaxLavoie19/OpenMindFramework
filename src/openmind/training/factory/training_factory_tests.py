from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.training.factory.training_factory import (
    create_value_distiller,
    create_value_training_loop,
)
from openmind.training.service.value_distiller import ValueDistiller
from openmind.training.service.value_training_loop import ValueTrainingLoop


def test_create_value_distiller_gives_a_value_distiller(knowledge: KnowledgeBase) -> None:
    assert isinstance(create_value_distiller(knowledge), ValueDistiller)


def test_create_value_training_loop_gives_a_value_training_loop(knowledge: KnowledgeBase) -> None:
    assert isinstance(create_value_training_loop(knowledge), ValueTrainingLoop)
