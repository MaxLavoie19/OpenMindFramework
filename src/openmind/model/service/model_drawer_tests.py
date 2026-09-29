import math
import random
from pathlib import Path

import pytest

from openmind.inference.constant.certainty_constant import ACCURACY, SCORED
from openmind.knowledge.constant.task_constant import POSITION_VALUE, SIMULATION
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.constant.model_constant import RULES
from openmind.model.factory.model_factory import create_model_drawer, create_model_registry
from openmind.model.service.model_drawer import ModelDrawer

pytestmark = pytest.mark.log_level("INFO")


def base(tmp_path: Path) -> KnowledgeBase:
    return create_knowledge_base("models", tmp_path)


def heuristic(knowledge: KnowledgeBase, name: str) -> ModelRecord:
    """One candidate heuristic registered as a model of what a position is worth."""
    return create_model_registry().register(
        knowledge,
        ModelRecord(
            name,
            (POSITION_VALUE,),
            knowledge.ensure_context("chess").id,
            RULES,
            knowledge.ensure_mechanism(name).id,
            f"ruleset-{name}",
        ),
    )


def scored(knowledge: KnowledgeBase, model: ModelRecord, accuracy: float, games: int) -> None:
    """What that model has been measured at, over that many games."""
    knowledge.believe(
        Belief(ACCURACY.format(mechanism=model.mechanism), model.context, accuracy, tags=((SCORED, games),))
    )


def test_a_model_that_has_never_played_is_drawn_before_one_that_has(tmp_path: Path) -> None:
    """A model never played has everything to prove, so it goes first. That is not a trick: an infinite bound
    is what 'no evidence' means on this scale, and it is why a pool that grows keeps being explored rather
    than settling on whatever happened to be registered early."""
    knowledge = base(tmp_path)
    context = knowledge.ensure_context("chess").id
    played = heuristic(knowledge, "played")
    scored(knowledge, played, accuracy=0.9, games=50)
    heuristic(knowledge, "never played")

    drawn = create_model_drawer().drawn(knowledge, context, POSITION_VALUE, random.Random(1))

    assert [one.name for one in drawn] == ["never played"]


def test_two_sides_are_drawn_two_different_models(tmp_path: Path) -> None:
    """One heuristic for white and a different one for black. A game between a model and itself settles
    nothing about either, so the draw is without replacement."""
    knowledge = base(tmp_path)
    context = knowledge.ensure_context("chess").id
    for name in ("one", "another", "a third"):
        scored(knowledge, heuristic(knowledge, name), accuracy=0.5, games=10)
    drawer = create_model_drawer()

    for seed in range(20):
        drawn = drawer.drawn(knowledge, context, POSITION_VALUE, random.Random(seed), how_many=2)

        assert len(drawn) == 2
        assert drawn[0].id != drawn[1].id


def test_a_pool_smaller_than_what_was_asked_for_gives_what_it_has(tmp_path: Path) -> None:
    """A caller wanting two and given one is being told something true about the pool, so this is not an
    error to raise about."""
    knowledge = base(tmp_path)
    context = knowledge.ensure_context("chess").id
    scored(knowledge, heuristic(knowledge, "the only one"), accuracy=0.5, games=10)

    drawn = create_model_drawer().drawn(knowledge, context, POSITION_VALUE, random.Random(1), how_many=2)

    assert [one.name for one in drawn] == ["the only one"]


def test_a_task_nothing_fills_draws_nothing(tmp_path: Path) -> None:
    knowledge = base(tmp_path)
    context = knowledge.ensure_context("chess").id
    heuristic(knowledge, "a position valuer")

    assert create_model_drawer().drawn(knowledge, context, SIMULATION, random.Random(1)) == ()


def test_what_has_done_better_is_drawn_more_often_where_all_have_played(tmp_path: Path) -> None:
    """Drawn rather than taken: what is worth trying is tried often and the rest is tried sometimes. Taken as
    the maximum every time, a heuristic that lost its opening games could never be shown to be good."""
    knowledge = base(tmp_path)
    context = knowledge.ensure_context("chess").id
    scored(knowledge, heuristic(knowledge, "better"), accuracy=0.9, games=30)
    scored(knowledge, heuristic(knowledge, "worse"), accuracy=0.1, games=30)
    drawer = create_model_drawer()

    counted = {"better": 0, "worse": 0}
    for seed in range(400):
        counted[drawer.drawn(knowledge, context, POSITION_VALUE, random.Random(seed))[0].name] += 1

    assert counted["better"] > counted["worse"]
    assert counted["worse"] > 0, "the worse one is still tried, which is the whole point of drawing"


def test_what_is_not_yet_known_about_a_model_is_what_the_bound_adds(tmp_path: Path) -> None:
    """Infinite where it has never been scored; otherwise what it has done plus room for what is not known,
    and that room shrinks as it plays more."""
    knowledge = base(tmp_path)
    unproven, little, lots = (heuristic(knowledge, name) for name in ("unproven", "little", "lots"))
    scored(knowledge, little, accuracy=0.5, games=4)
    scored(knowledge, lots, accuracy=0.5, games=400)
    drawer = create_model_drawer()

    assert drawer.bound(knowledge, unproven, among=3) == math.inf
    assert drawer.bound(knowledge, little, among=3) > drawer.bound(knowledge, lots, among=3)
    assert drawer.bound(knowledge, lots, among=3) > 0.5, "what it has done, plus something"


def test_a_drawer_without_a_registry_says_so(tmp_path: Path) -> None:
    """Rather than quietly drawing nothing, which would look like a task nothing fills."""
    knowledge = base(tmp_path)

    with pytest.raises(ValueError, match="registry"):
        ModelDrawer().drawn(knowledge, knowledge.ensure_context("chess").id, POSITION_VALUE, random.Random(1))
