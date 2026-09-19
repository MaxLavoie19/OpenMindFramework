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
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT, EFFECTS
from openmind.testing.plugin.game_fixtures import simulation_rules
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.players import Players
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


def test_the_standard_game_starts_with_nine_empty_cells() -> None:
    assert create_tictactoe_initial_state() == State.of(
        cell=Grid.filled((3, 3), None), turn="X", payoff=Map.of({"X": None, "O": None})
    )


def test_a_mark_is_placed_on_an_empty_cell_while_no_payoff_is_set(game: Game) -> None:
    rbs = game("tictactoe")
    marked = rbs.outcomes(rbs.start(), rbs.actions(rbs.start())[0]).outcomes[0][0]

    assert {dict(action.parameters)["row"] for action in rbs.actions(rbs.start())} == {1, 2, 3}
    assert len(rbs.actions(marked)) == 8


def test_fourinarow_starts_with_42_empty_cells() -> None:
    assert create_tictactoe_initial_state(VARIANTS["fourinarow"]) == State.of(
        cell=Grid.filled((6, 7), None), turn="X", payoff=Map.of({"X": None, "O": None})
    )


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

    assert RuleRunner(StateNamespaceMapper()).value(reading, State(())) == expected


@pytest.mark.parametrize(("name", "action"), [("standard", "place"), ("fourinarow", "drop"), ("gomoku", "place")])
def test_each_variant_s_move_leads_somewhere_certain(knowledge: KnowledgeBase, name: str, action: str) -> None:
    context = declare_tictactoe(knowledge, VARIANTS[name])

    (effects,) = simulation_rules(knowledge, context, (EFFECTS,))

    assert (effects.action, effects.probability) == (action, 1.0)


def test_players_are_x_and_o_with_the_payoff_map() -> None:
    assert create_tictactoe_players() == Players(("X", "O"), "payoff")


def test_the_declared_game_holds_the_rules_of_tic_tac_toe(knowledge: KnowledgeBase) -> None:
    context = declare_tictactoe(knowledge)

    assert context == "tictactoe"
    assert [rule.action for rule in simulation_rules(knowledge, context, (CONSTRAINT,))] == ["place"] * 4


def test_only_the_player_whose_turn_it_is_has_an_action(game: Game) -> None:
    rbs = game("tictactoe")
    marked = rbs.outcomes(rbs.start(), rbs.actions(rbs.start())[0]).outcomes[0][0]

    assert (rbs.acting(rbs.start()), rbs.acting(marked)) == ((0,), (1,))


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
