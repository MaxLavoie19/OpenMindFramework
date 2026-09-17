from dataclasses import replace

import pytest

from openmind.agent.constant.tictactoe_constant import STANDARD, VARIANTS
from openmind.agent.factory.tictactoe_factory import (
    create_tictactoe_definitions,
    create_tictactoe_domain,
    create_tictactoe_initial_state,
    create_tictactoe_players,
    create_tictactoe_timeout,
    create_tictactoe_problem,
    create_tictactoe_transitions,
)
from openmind.agent.model.domain import Domain
from openmind.agent.model.tictactoe_variant import TicTacToeVariant
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State

NO_PAYOFF_SET = (PythonRule("payoff['X'] is None"), PythonRule("payoff['O'] is None"))


def test_initial_state_has_empty_cells_x_to_play_and_no_payoff() -> None:
    variables = dict(create_tictactoe_initial_state().variables)

    assert variables == {
        **{f"cell({row},{col})": None for row in (1, 2, 3) for col in (1, 2, 3)},
        "turn": "X",
        "payoff(X)": None,
        "payoff(O)": None,
    }


def test_problem_places_a_mark_on_an_empty_cell_while_no_payoff_is_set() -> None:
    positions = DiscreteDomain((1, 2, 3))

    assert create_tictactoe_problem() == Problem(
        (
            ActionDefinition(
                "place",
                (Variable("row", positions), Variable("col", positions)),
                (*NO_PAYOFF_SET, PythonRule("cell[row, col] is None")),
            ),
        ),
        create_tictactoe_definitions(),
    )


def test_fourinarow_starts_with_42_empty_cells() -> None:
    variables = dict(create_tictactoe_initial_state(VARIANTS["fourinarow"]).variables)

    assert variables == {
        **{f"cell({row},{col})": None for row in range(1, 7) for col in range(1, 8)},
        "turn": "X",
        "payoff(X)": None,
        "payoff(O)": None,
    }


def test_fourinarow_drops_a_mark_in_a_column_whose_top_cell_is_empty() -> None:
    assert create_tictactoe_problem(VARIANTS["fourinarow"]) == Problem(
        (
            ActionDefinition(
                "drop",
                (Variable("col", DiscreteDomain((1, 2, 3, 4, 5, 6, 7))),),
                (*NO_PAYOFF_SET, PythonRule("cell[1, col] is None")),
            ),
        ),
        create_tictactoe_definitions(VARIANTS["fourinarow"]),
    )


@pytest.mark.parametrize(("name", "expected"), [("standard", (3, 3, 3, 4)), ("fourinarow", (7, 6, 4, 13)), ("gomoku", (15, 15, 5, 20))])
def test_definitions_give_the_sizes_and_the_lines_through_the_centre(name: str, expected: tuple[int, ...]) -> None:
    reading = RuleCompiler().compile_value(
        PythonRule("(WIDTH, HEIGHT, LINE, len(LINES_THROUGH[(HEIGHT + 1) // 2, (WIDTH + 1) // 2]))"),
        (),
        create_tictactoe_definitions(VARIANTS[name]),
    )

    assert RuleRunner(StateNamespaceMapper(VariableNameMapper())).value(reading, State(())) == expected


@pytest.mark.parametrize(("name", "action"), [("standard", "place"), ("fourinarow", "drop"), ("gomoku", "place")])
def test_transitions_give_the_variant_action_one_certain_branch(name: str, action: str) -> None:
    (transition,) = create_tictactoe_transitions(VARIANTS[name]).transitions
    (branch,) = transition.branches

    assert (transition.action, branch.probability) == (action, 1.0)


def test_players_are_x_and_o_with_turn_and_payoff_variables() -> None:
    assert create_tictactoe_players() == Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)"))


def test_domain_holds_the_tictactoe_recipes() -> None:
    assert create_tictactoe_domain() == Domain(
        "tictactoe",
        create_tictactoe_initial_state(),
        create_tictactoe_problem(),
        create_tictactoe_transitions(),
        create_tictactoe_players(),
        timeout=create_tictactoe_timeout(),
    )


def test_a_variant_is_named_after_the_game_and_the_standard_game_keeps_its_name() -> None:
    fourinarow = VARIANTS["fourinarow"]

    assert create_tictactoe_domain(fourinarow) == Domain(
        "tictactoe/fourinarow",
        create_tictactoe_initial_state(fourinarow),
        create_tictactoe_problem(fourinarow),
        create_tictactoe_transitions(fourinarow),
        create_tictactoe_players(),
        timeout=create_tictactoe_timeout(),
    )
    assert create_tictactoe_domain(VARIANTS["standard"]).name == "tictactoe"


@pytest.mark.parametrize(
    "variant",
    [replace(STANDARD, width=0), replace(STANDARD, height=0), replace(STANDARD, line=0), replace(STANDARD, line=4)],
)
def test_a_variant_without_room_for_its_line_raises(variant: TicTacToeVariant) -> None:
    with pytest.raises(ValueError, match="A tic-tac-toe variant needs a width and a height of at least 1"):
        create_tictactoe_domain(variant)
