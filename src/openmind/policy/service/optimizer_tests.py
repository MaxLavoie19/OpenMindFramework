import random
from collections.abc import Callable

from openmind.game.service.game_declarer import GameDeclarer
from openmind.knowledge.constant.rule_kind_constant import OPTIMUM
from openmind.knowledge.constant.task_constant import OPTIMIZING
from openmind.knowledge.model.policy import Policy
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.policy.factory.policy_factory import create_random_picker, create_rule_optimizer, create_solver_optimizer
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_rule_based_system
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]

MIDDLE = Action("place", (("col", 2), ("row", 2)))


def flee(knowledge: KnowledgeBase, context: str = "tictactoe") -> Policy:
    return Policy("flee", knowledge.ensure_context(context).id, "take the middle")


def test_the_random_picker_takes_one_of_the_legal_actions(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())

    chosen = create_random_picker(random.Random(1)).optimize(None, flee(knowledge), node, "X")

    assert chosen in played.actions(played.start(), player="X")


def test_the_random_picker_gives_nothing_where_the_player_has_no_action(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())

    assert create_random_picker(random.Random(1)).optimize(None, flee(knowledge), node, "O") is None


def test_the_solver_optimizer_settles_on_the_first_action_it_works_out(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())

    chosen = create_solver_optimizer().optimize(None, flee(knowledge), node, "X")

    assert chosen == Action("place", (("col", 1), ("row", 1)))
    assert create_solver_optimizer().optimize(None, flee(knowledge), node, "O") is None


def optimizing(knowledge: KnowledgeBase, rule: str, context: str = "aiming") -> object:
    declarer = GameDeclarer(knowledge, context, ruleset="optimizing")
    declarer.rule("what to play", OPTIMUM, PythonRule(rule))
    declarer.done()
    ruleset = knowledge.ruleset_named(knowledge.context_named(context).id, "optimizing")  # type: ignore[union-attr]
    from dataclasses import replace

    knowledge.ruleset(replace(ruleset, task=OPTIMIZING))  # type: ignore[arg-type]
    return create_rule_based_system(knowledge, context, OPTIMIZING)


def test_the_rule_optimizer_computes_the_action_its_rules_say(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    rules = optimizing(knowledge, "('place', {'row': 2, 'col': 2})")

    assert create_rule_optimizer().optimize(rules, flee(knowledge), node, "X") == MIDDLE


def test_an_action_the_constraints_refuse_is_no_solution(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    off_the_board = optimizing(knowledge, "('place', {'row': 9, 'col': 9})", "aiming badly")

    assert create_rule_optimizer().optimize(off_the_board, flee(knowledge), node, "X") is None


def test_an_optimum_rule_reads_the_policy_s_sub_goal_and_the_player(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    rules = optimizing(knowledge, "('place', {'row': 2, 'col': 2}) if sub_goal == 'take the middle' and player == 'X' else None", "aiming at the middle")

    assert create_rule_optimizer().optimize(rules, flee(knowledge), node, "X") == MIDDLE
    assert create_rule_optimizer().optimize(rules, flee(knowledge), node, "O") is None
