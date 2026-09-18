from collections.abc import Callable

from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system, create_value_generator
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.service.value_generator import ValueGenerator

type Game = Callable[[str], RuleBasedSystem]


def test_create_rule_based_system_holds_the_rules_of_its_context(game: Game) -> None:
    rbs = game("tictactoe")

    assert rbs.context == "tictactoe"
    assert len(rbs.actions(rbs.start())) == 9


def test_a_context_with_position_rules_values_a_position_with_them(
    game: Game, knowledge: KnowledgeBase
) -> None:
    rbs = game("tictactoe")
    declarer = RuleDeclarer(knowledge, rbs.context)
    declarer.position("a mark on the middle cell", PythonRule("cell[2, 2] == me"), 0.75)
    declarer.position("a constant", PythonRule("1.0"), 0.25)

    valued = create_rule_based_system(knowledge, rbs.context)
    middle = valued.outcomes(valued.start(), valued.actions(valued.start())[4]).outcomes[0][0]

    assert valued.value(valued.start(), "X") == 0.25
    assert valued.value(middle, "X") == 1.0


def test_create_value_generator_gives_a_value_generator() -> None:
    assert isinstance(create_value_generator(), ValueGenerator)
