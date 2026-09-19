from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.training.builder.value_distiller_builder import ValueDistillerBuilder
from openmind.training.service.value_distiller import ValueDistiller


def test_build_gives_a_value_distiller(knowledge: KnowledgeBase) -> None:
    assert isinstance(
        ValueDistillerBuilder().with_workers(2).with_knowledge_base(knowledge).build(), ValueDistiller
    )


def test_a_distiller_without_a_knowledge_base_is_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="needs a knowledge base"):
        ValueDistillerBuilder().with_workers(2).build()
