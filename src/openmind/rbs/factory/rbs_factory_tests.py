from collections.abc import Callable

from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_value_generator
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.value_generator import ValueGenerator

type Game = Callable[[str], RuleBasedGame]


def test_create_rule_based_system_holds_the_rules_of_its_context(game: Game) -> None:
    rbs = game("tictactoe")

    assert rbs.context == "tictactoe"
    assert len(rbs.actions(rbs.start())) == 9


def test_a_context_with_position_rules_values_a_position_with_them(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    rbs = game("tictactoe")
    heuristic(rbs.context, "a mark on the middle cell", PythonRule("cell[2, 2] == me"), 0.75)
    heuristic(rbs.context, "a constant", PythonRule("1.0"), 0.25)

    valued = create_rule_based_game(knowledge, rbs.context)
    middle = valued.outcomes(valued.start(), valued.actions(valued.start())[4]).outcomes[0][0]

    assert valued.value(valued.start(), "X") == 0.25
    assert valued.value(middle, "X") == 1.0


def test_create_value_generator_gives_a_value_generator() -> None:
    assert isinstance(create_value_generator(), ValueGenerator)


def test_a_context_without_a_ruleset_of_its_own_for_a_task_takes_the_one_it_inherits(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    from dataclasses import replace

    rbs = game("tictactoe")
    heuristic("a round", "a constant", PythonRule("1.0"), 0.5)
    round_context = knowledge.context_named("a round")
    knowledge.context(replace(round_context, inherits=(rbs.context_id,)))  # type: ignore[arg-type]

    valued = create_rule_based_game(knowledge, "a round")

    assert len(valued.actions(valued.start())) == 9
    assert valued.value(valued.start(), "X") == 0.5


def test_the_rbs_of_a_context_s_ruleset_for_a_task_is_that_ruleset_s_rules_at_their_weights(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    from openmind.knowledge.constant.knowledge_constant import POSITION_VALUE, SIMULATION
    from openmind.rbs.factory.rbs_factory import create_rule_based_system

    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.25)

    simulation = create_rule_based_system(knowledge, "tictactoe")
    position = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)

    assert (simulation.ruleset.task, position.ruleset.task) == (SIMULATION, POSITION_VALUE)
    assert [(rule.name, weight) for rule, weight in position.rules] == [("a constant", 0.25)]


def test_a_context_without_a_ruleset_for_the_task_has_no_rbs_for_it(knowledge: KnowledgeBase) -> None:
    import pytest

    from openmind.rbs.factory.rbs_factory import create_rule_based_system

    knowledge.ensure_context("nothing")

    with pytest.raises(ValueError, match="nothing has no simulation ruleset"):
        create_rule_based_system(knowledge, "nothing")
