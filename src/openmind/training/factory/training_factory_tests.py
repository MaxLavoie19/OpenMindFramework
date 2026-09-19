from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.service.value_distiller import ValueDistiller


def test_create_value_distiller_gives_a_value_distiller(knowledge: KnowledgeBase) -> None:
    assert isinstance(create_value_distiller(knowledge), ValueDistiller)
