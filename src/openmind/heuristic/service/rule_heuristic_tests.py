from collections.abc import Callable

from openmind.knowledge.constant.knowledge_constant import INFERENCE
from openmind.knowledge.constant.rule_kind_constant import MOVE, POSITION
from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_rule_based_system, create_rule_heuristic
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]


def test_a_position_is_worth_its_rules_readings_times_their_weights_summed(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "a mark on the middle cell", PythonRule("cell[2, 2] == me"), 0.75)
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.25)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    reader = create_rule_heuristic()
    game_played = create_rule_based_game(knowledge, "tictactoe")
    start = game_played.node(game_played.start())

    assert reader.value(rbs, start, "X") == 0.25
    assert reader.values(rbs, start, ("X", "O")) == (0.25, 0.25)
    assert [rule.name for rule, _ in reader.explain(rbs, start, "X")] == ["a mark on the middle cell", "a constant"]


def test_a_rule_that_cant_be_read_here_is_left_out_and_one_reading_nothing_adds_nothing(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "reads a model the game has not", PythonRule("lamp"), 1.0)
    heuristic("tictactoe", "reads nothing here", PythonRule("None"), 1.0)
    heuristic("tictactoe", "a constant", PythonRule("2.0"), 0.5)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")
    start = played.node(played.start())

    added = dict(create_rule_heuristic().explain(rbs, start, "X"))

    assert [rule.name for rule in added] == ["reads nothing here", "a constant"]
    assert create_rule_heuristic().value(rbs, start, "X") == 1.0


def test_a_move_is_rated_for_the_player_taking_it_and_its_parameters_are_read(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "the middle cell is worth taking", PythonRule("row == 2 and col == 2"), 1.0, MOVE)
    rbs = create_rule_based_system(knowledge, "tictactoe", MOVE_VALUE)
    facade = create_rule_based_game(knowledge, "tictactoe")
    actions = (Action("place", (("col", 1), ("row", 1))), Action("place", (("col", 2), ("row", 2))))

    assert create_rule_heuristic().rate(rbs, facade.node(facade.start()), actions, "X") == (0.0, 1.0)


def test_a_ruleset_with_no_rule_of_that_kind_says_nothing(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 1.0)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")
    start = played.node(played.start())

    assert create_rule_heuristic().rate(rbs, start, (Action("place", (("col", 1), ("row", 1))),), "X") == (None,)


def test_two_rulesets_judging_alike_describe_alike(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.5)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)

    described = create_rule_heuristic().describe(rbs)

    assert '"a constant"' in described and "0.5" in described


def test_a_feature_is_extracted_once_and_shared_by_every_model_reading_the_node(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "my winning moves", PythonRule("wins(me)"), 1.0)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")
    node = played.node(played.start())
    reader = create_rule_heuristic()

    reader.value(rbs, node, "X")
    reader.value(rbs, node, "X")

    assert list(node.features) == ["consequences of X"]
