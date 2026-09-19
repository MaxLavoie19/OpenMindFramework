from pathlib import Path

import pytest

from openmind.knowledge.constant.knowledge_constant import DECLARATION, INFERENCE
from openmind.knowledge.constant.task_constant import POSITION_VALUE, SIMULATION
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.constant.model_constant import NETWORK, RULES
from openmind.model.factory.model_factory import create_model_registry, create_model_timer

pytestmark = pytest.mark.log_level("INFO")


def base(tmp_path: Path) -> KnowledgeBase:
    return create_knowledge_base("models", tmp_path)


def network(knowledge: KnowledgeBase, name: str, task: str = POSITION_VALUE, context: str = "chess") -> ModelRecord:
    return create_model_registry().register(
        knowledge,
        ModelRecord(
            name,
            task,
            knowledge.ensure_context(context).id,
            NETWORK,
            knowledge.ensure_mechanism(name).id,
            f"data/model/{name}.pt",
        ),
    )


def test_a_model_is_kept_with_its_task_its_family_and_where_its_data_lives(tmp_path: Path) -> None:
    knowledge = base(tmp_path)

    kept = network(knowledge, "the 2026-09 network")

    assert kept.id.startswith("model-")
    assert (kept.task, kept.family, kept.location) == (POSITION_VALUE, NETWORK, "data/model/the 2026-09 network.pt")
    assert knowledge.model_by_id(kept.id) == kept
    assert create_knowledge_base("models", tmp_path).model_by_id(kept.id) == kept


def test_registering_the_same_name_in_the_same_context_again_updates_it(tmp_path: Path) -> None:
    knowledge = base(tmp_path)
    first = network(knowledge, "the 2026-09 network")

    again = network(knowledge, "the 2026-09 network")

    assert again.id == first.id
    assert len(knowledge.models()) == 1


def test_a_ruleset_is_registered_as_a_model_of_its_task_found_again_by_its_id(tmp_path: Path) -> None:
    knowledge = base(tmp_path)
    context = knowledge.ensure_context("chess").id
    ruleset = knowledge.ruleset(
        Ruleset("simulation", context, SIMULATION, Source(knowledge.ensure_mechanism(DECLARATION).id))
    )

    model = create_model_registry().register_ruleset(knowledge, ruleset)

    assert (model.family, model.task, model.location) == (RULES, SIMULATION, ruleset.id)


def test_the_models_of_a_task_come_from_the_context_or_from_the_one_it_inherits(tmp_path: Path) -> None:
    from dataclasses import replace

    knowledge = base(tmp_path)
    registry = create_model_registry()
    network(knowledge, "the 2026-09 network")
    variant = knowledge.ensure_context("chess/960")
    knowledge.context(replace(variant, inherits=(knowledge.ensure_context("chess").id,)))

    inherited = registry.of_task(knowledge, variant.id, POSITION_VALUE)

    assert [model.name for model in inherited] == ["the 2026-09 network"]
    assert registry.of_task(knowledge, variant.id, SIMULATION) == ()


def test_the_best_model_of_a_task_is_the_one_measured_most_accurate_ties_to_the_fastest(tmp_path: Path) -> None:
    knowledge = base(tmp_path)
    registry, timer = create_model_registry(), create_model_timer()
    context = knowledge.ensure_context("chess").id
    accurate, quick, unmeasured = (network(knowledge, name) for name in ("accurate", "quick", "unmeasured"))
    for model, accuracy in ((accurate, 0.9), (quick, 0.6)):
        knowledge.believe(Belief(f"accuracy of {model.mechanism}", context, accuracy, tags=(("scored", 4),)))
    timer.spent(knowledge, accurate, 0.5)
    timer.spent(knowledge, quick, 0.01)

    assert registry.best(knowledge, context, POSITION_VALUE) == accurate
    assert registry.measured(knowledge, accurate).accuracy == pytest.approx(0.9)
    assert registry.measured(knowledge, unmeasured).accuracy is None
    assert registry.best(knowledge, context, SIMULATION) is None


def test_a_model_of_no_task_here_has_no_best(tmp_path: Path) -> None:
    knowledge = base(tmp_path)

    assert create_model_registry().best(knowledge, knowledge.ensure_context("chess").id, POSITION_VALUE) is None
