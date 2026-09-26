from collections.abc import Callable

from openmind.heuristic.model.move_rater import MoveRater
from openmind.heuristic.service.rule_move_rater import RuleMoveRater
from openmind.knowledge.constant.rule_kind_constant import MOVE
from openmind.knowledge.constant.task_constant import MOVE_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_rule_based_system, create_rule_heuristic
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]

MIDDLE = Action("place", (("col", 2), ("row", 2)))
CORNER = Action("place", (("col", 1), ("row", 1)))


def test_each_action_is_worth_what_the_move_rules_read_of_it(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "the middle cell is worth taking", PythonRule("row == 2 and col == 2"), 4.0, MOVE)
    rbs = create_rule_based_system(knowledge, "tictactoe", MOVE_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")

    rated = RuleMoveRater(create_rule_heuristic()).rate(rbs, played.node(played.start()), (CORNER, MIDDLE), "X")

    assert rated == (0.0, 4.0)


def test_a_model_with_no_move_rules_knows_nothing_about_any_action(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """None is "knows nothing", not zero, the whole way through: a model without a move rule has no opinion of an
    action, which is a different thing from thinking it worthless."""
    game("tictactoe")
    heuristic("tictactoe", "a position rule and no move rule", PythonRule("1.0"), 1.0, task=MOVE_VALUE)
    rbs = create_rule_based_system(knowledge, "tictactoe", MOVE_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")

    rated = RuleMoveRater(create_rule_heuristic()).rate(rbs, played.node(played.start()), (CORNER, MIDDLE), "X")

    assert rated == (None, None)


def test_it_is_the_port_and_the_service_it_wraps_is_not(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """`RuleHeuristic` satisfies the port by an accident of arity — one of its six methods happens to have the
    signature. This pins that the rater is what fills the port, so a second family of model sits beside it
    rather than having to displace the rules service."""
    game("tictactoe")
    heuristic("tictactoe", "the middle cell is worth taking", PythonRule("row == 2 and col == 2"), 1.0, MOVE)
    rbs = create_rule_based_system(knowledge, "tictactoe", MOVE_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")
    node = played.node(played.start())
    rater: MoveRater[object] = RuleMoveRater(create_rule_heuristic())

    assert rater.rate(rbs, node, (MIDDLE,), "X") == create_rule_heuristic().rate(rbs, node, (MIDDLE,), "X")
