from collections.abc import Callable
from dataclasses import replace

import pytest

from openmind.agent.constant.tictactoe_constant import STANDARD, VARIANTS
from openmind.agent.factory.tictactoe_factory import (
    create_tictactoe_definitions,
    create_tictactoe_initial_state,
    create_tictactoe_players,
    declare_tictactoe,
)
from openmind.agent.model.tictactoe_variant import TicTacToeVariant
from openmind.doxastic.constant.rule_kind_constant import CONSTRAINT, EFFECTS, TIMEOUT
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedSystem]


def test_the_standard_game_starts_with_nine_empty_cells() -> None:
    variables = dict(create_tictactoe_initial_state().variables)

    assert variables == {
        **{f"cell({row},{col})": None for row in range(1, 4) for col in range(1, 4)},
        "turn": "X",
        "payoff(X)": None,
        "payoff(O)": None,
    }


def test_a_mark_is_placed_on_an_empty_cell_while_no_payoff_is_set(game: Game) -> None:
    rbs = game("tictactoe")
    marked = rbs.outcomes(rbs.start(), rbs.actions(rbs.start())[0]).outcomes[0][0]

    assert {dict(action.parameters)["row"] for action in rbs.actions(rbs.start())} == {1, 2, 3}
    assert len(rbs.actions(marked)) == 8


def test_fourinarow_starts_with_42_empty_cells() -> None:
    variables = dict(create_tictactoe_initial_state(VARIANTS["fourinarow"]).variables)

    assert variables == {
        **{f"cell({row},{col})": None for row in range(1, 7) for col in range(1, 8)},
        "turn": "X",
        "payoff(X)": None,
        "payoff(O)": None,
    }


def test_fourinarow_drops_a_mark_in_a_column_whose_top_cell_is_empty(game: Game) -> None:
    rbs = game("tictactoe/fourinarow")

    actions = rbs.actions(rbs.start())

    assert {action.name for action in actions} == {"drop"}
    assert sorted(dict(action.parameters)["col"] for action in actions) == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize(("name", "expected"), [("standard", (3, 3, 3, 4)), ("fourinarow", (7, 6, 4, 13)), ("gomoku", (15, 15, 5, 20))])
def test_definitions_give_the_sizes_and_the_lines_through_the_centre(name: str, expected: tuple[int, ...]) -> None:
    reading = RuleCompiler().compile_value(
        PythonRule("(WIDTH, HEIGHT, LINE, len(LINES_THROUGH[(HEIGHT + 1) // 2, (WIDTH + 1) // 2]))"),
        (),
        create_tictactoe_definitions(VARIANTS[name]),
    )

    assert RuleRunner(StateNamespaceMapper(VariableNameMapper())).value(reading, State(())) == expected


@pytest.mark.parametrize(("name", "action"), [("standard", "place"), ("fourinarow", "drop"), ("gomoku", "place")])
def test_each_variant_s_move_leads_somewhere_certain(knowledge: KnowledgeBase, name: str, action: str) -> None:
    context = declare_tictactoe(knowledge, VARIANTS[name])

    (effects,) = knowledge.rules(context, (EFFECTS,))

    assert (effects.action, effects.probability) == (action, 1.0)


def test_players_are_x_and_o_with_turn_and_payoff_variables() -> None:
    assert create_tictactoe_players() == Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)"))


def test_the_declared_game_holds_the_rules_of_tic_tac_toe(knowledge: KnowledgeBase) -> None:
    context = declare_tictactoe(knowledge)

    assert context == "tictactoe"
    assert [rule.action for rule in knowledge.rules(context, (CONSTRAINT,))] == ["place", "place", "place"]
    assert [rule.kind for rule in knowledge.rules(context, (TIMEOUT,))] == [TIMEOUT]


def test_a_variant_is_declared_under_its_own_name_and_the_standard_game_keeps_its_name(knowledge: KnowledgeBase) -> None:
    assert declare_tictactoe(knowledge, VARIANTS["fourinarow"]) == "tictactoe/fourinarow"
    assert declare_tictactoe(knowledge, VARIANTS["standard"]) == "tictactoe"


@pytest.mark.parametrize(
    "variant",
    [replace(STANDARD, width=0), replace(STANDARD, height=0), replace(STANDARD, line=0), replace(STANDARD, line=4)],
)
def test_a_variant_without_room_for_its_line_raises(knowledge: KnowledgeBase, variant: TicTacToeVariant) -> None:
    with pytest.raises(ValueError, match="A tic-tac-toe variant needs a width and a height of at least 1"):
        declare_tictactoe(knowledge, variant)
