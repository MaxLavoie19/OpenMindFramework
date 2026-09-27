from collections.abc import Callable

from openmind.heuristic.model.node import Node
from openmind.heuristic.service.rule_position_valuer import RulePositionValuer
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_rule_based_system, create_rule_heuristic
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


def valuing(knowledge: KnowledgeBase, named: str = "tictactoe"):
    played = create_rule_based_game(knowledge, named)
    return (
        RulePositionValuer(create_rule_heuristic()),
        create_rule_based_system(knowledge, named, POSITION_VALUE),
        played.node(played.start()),
    )


def test_the_players_come_from_the_node_s_game_rather_than_from_the_valuer(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """The port takes a node and nothing else, so who the players are has to be read off the game it carries.
    Two of them, in the order the game declared them, and one value each."""
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("2.0"), 0.5)
    valuer, rbs, start = valuing(knowledge)

    assert valuer.values(rbs, start) == (1.0, 1.0)
    assert len(start.game.players().names) == 2  # type: ignore[union-attr]


def test_a_position_no_rule_can_be_read_in_is_worth_nothing_known(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """None means it knows nothing here, not that the position is worth zero — the difference a planner needs
    to fall back on something else rather than believe a nought."""
    game("tictactoe")
    heuristic("tictactoe", "reads a model the game has not", PythonRule("lamp"), 1.0)
    valuer, rbs, start = valuing(knowledge)

    assert valuer.values(rbs, start) is None


def test_what_it_judges_with_is_readable_rule_by_rule(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """A game remembers what each side played with by asking the service that ran it, so what won can be read
    afterwards rather than only named."""
    game("tictactoe")
    heuristic("tictactoe", "a mark on the middle cell", PythonRule("cell[2, 2] == me"), 0.75)
    valuer, rbs, _ = valuing(knowledge)

    said = valuer.describe(rbs)

    assert "a mark on the middle cell" in said
    assert "0.75" in said


def test_two_models_judging_alike_describe_alike(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """Which is what gives them the same id. A heuristic that changed a weight is a different heuristic."""
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("2.0"), 0.5)
    valuer, rbs, _ = valuing(knowledge)
    again = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)

    assert valuer.describe(rbs) == valuer.describe(again)
