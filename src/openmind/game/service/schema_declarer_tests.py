from openmind.game.service.game_declarer import GameDeclarer
from openmind.game.service.schema_declarer import SchemaDeclarer
from openmind.knowledge.constant.rule_kind_constant import VALUES
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.structure.model.domain import Numbers
from openmind.structure.model.kind import Kind
from openmind.structure.model.schema import ActionKind, Schema
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.world.model.state import State


def a_schema() -> Schema:
    """A game whose move takes a mark and a step, the step left unbounded as a real game leaves it."""
    return Schema(
        models=(),
        actions=(
            ActionKind("move", (("mark", Kind("mark", values=("x", "o"))), ("step", Numbers(whole=True, least=-2, most=2)))),
        ),
    )


def declaring(knowledge: KnowledgeBase) -> GameDeclarer:
    return GameDeclarer(knowledge, "a little game", rule_caller=create_rule_caller(), open=True)


def test_every_parameter_of_an_action_becomes_a_rule_saying_what_it_can_be(knowledge: KnowledgeBase) -> None:
    """The solver enumerates from the domains and the constraints only narrow what was enumerated, so a game
    with no values rule can list no move at all — however much it has learned about legality."""
    declarer = declaring(knowledge)

    declared = SchemaDeclarer().declare(declarer, a_schema(), "move")

    assert len(declared) == 2
    assert all(one.kind == VALUES for one in declared)
    assert {one.parameter for one in declared} == {"mark", "step"}


def test_what_a_rule_gives_back_is_what_the_schema_said_the_parameter_ranges_over(knowledge: KnowledgeBase) -> None:
    """Written out rather than computed: what a parameter can be before anything narrows it does not depend on
    the position, so the rule needs no function a worker process would have to find."""
    declarer = declaring(knowledge)
    caller = create_rule_caller()

    declared = {one.parameter: one for one in SchemaDeclarer().declare(declarer, a_schema(), "move")}

    empty = State.of()
    assert caller.value(declared["mark"].rule, empty) == ["x", "o"]
    assert caller.value(declared["step"].rule, empty) == [-2, -1, 0, 1, 2]
