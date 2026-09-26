import json
from collections.abc import Callable

import pytest

from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE
from openmind.knowledge.constant.rule_kind_constant import MOVE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]

MIDDLE = Action("place", (("col", 2), ("row", 2)))


def test_a_context_with_no_simulation_says_what_is_missing_rather_than_failing_later(
    knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """A game with only a heuristic is a real thing to have — a context that judges positions of a game
    declared elsewhere — so building one is allowed and only asking it to play is not."""
    heuristic("a judged context", "a constant", PythonRule("1.0"), 1.0)

    played = create_rule_based_game(knowledge, "a judged context")

    with pytest.raises(ValueError, match="where the game starts"):
        played.simulation_rbs


def test_it_runs_every_rule_of_every_ruleset_it_was_given_with_the_simulation_first(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """A game is its simulation and whatever judges it, and something explaining a decision wants them in the
    order they were consulted."""
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.25)
    played = create_rule_based_game(knowledge, "tictactoe")

    kinds = [rule.kind for rule in played.rules]

    assert "position" in kinds
    assert kinds.index("initial") < kinds.index("position")


def test_a_rule_weighs_what_its_own_heuristic_lists_it_at(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    kept = heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.25)
    played = create_rule_based_game(knowledge, "tictactoe")

    assert played.weight(kept) == 0.25


def test_a_rule_no_heuristic_lists_weighs_nothing(game: Game, knowledge: KnowledgeBase) -> None:
    """A simulation rule is not a heuristic rule, so it must add nothing to a value rather than one."""
    played = game("tictactoe")

    assert played.weight(played.rules[0]) == 0.0


def test_a_game_with_no_position_heuristic_knows_nothing_rather_than_valuing_at_nought(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """None means knows nothing the whole way through: a search told a position is worth zero would treat it
    as a draw, which is a claim nobody made."""
    played = game("tictactoe")

    assert played.value(played.start(), "X") is None
    assert played.values(played.start()) is None
    assert played.explain(played.start(), "X") == ()


def test_a_game_with_no_move_heuristic_knows_nothing_about_any_action(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")

    assert played.rate(played.start(), (MIDDLE,)) == (None,)


def test_a_position_is_worth_what_its_heuristic_reads_and_a_move_what_its_own_reads(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """The two tasks are separate rulesets, and a game asked for one must not answer out of the other."""
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("2.0"), 0.5)
    heuristic("tictactoe", "the middle is worth taking", PythonRule("row == 2 and col == 2"), 4.0, MOVE)
    played = create_rule_based_game(knowledge, "tictactoe")

    assert played.value(played.start(), "X") == 1.0
    assert played.rate(played.start(), (MIDDLE,), "X") == (4.0,)


def test_a_game_that_does_not_draw_or_record_itself_says_nothing_rather_than_inventing_one(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """A picture and a record are the game's own business; a framework that made one up would be drawing a
    board it was told nothing about."""
    played = game("tictactoe")

    assert played.picture(played.start()) is None
    assert played.record(()) is None


def test_two_games_describe_alike_when_they_judge_alike(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """What a model is remembered by has to tell one heuristic from another and nothing else: a game whose
    context holds more rules of other tasks still judges the same."""
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.25)
    first = create_rule_based_game(knowledge, "tictactoe").describe()

    second = create_rule_based_game(knowledge, "tictactoe").describe()

    assert first == second
    assert json.loads(first)["position"][0][0] == "a constant"


def test_a_node_carries_the_game_so_what_a_heuristic_reads_is_worked_out_once(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """Every model asked about a position shares what was extracted from it, which is only possible because a
    node knows the game it belongs to."""
    played = game("tictactoe")

    node = played.node(played.start())

    assert node.game is played
