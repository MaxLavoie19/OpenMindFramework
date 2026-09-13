import pytest

from openmind.agent.factory.sudoku_factory import (
    create_sudoku_domain,
    create_sudoku_initial_state,
    create_sudoku_players,
    create_sudoku_problem,
    create_sudoku_transitions,
)
from openmind.agent.model.domain import Domain
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.variable import Variable
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.predictor.model.assign import Assign
from openmind.world.model.players import Players

TOP95_FIRST = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"


def test_initial_state_holds_the_clues_and_the_empty_cells() -> None:
    variables = dict(create_sudoku_initial_state().variables)

    assert (variables["cell(1,1)"], variables["cell(1,2)"], variables["cell(1,3)"], variables["cell(9,9)"]) == (
        5,
        3,
        None,
        9,
    )
    assert sum(1 for name, value in variables.items() if name.startswith("cell") and value is None) == 51
    assert (variables["turn"], variables["payoff"], len(variables)) == ("solver", None, 83)


def test_problem_fills_every_empty_cell_under_one_all_different_per_row_column_and_box() -> None:
    (fill,) = create_sudoku_problem().actions

    assert (fill.name, len(fill.variables), len(fill.constraints)) == ("fill", 51, 28)
    assert fill.variables[0] == Variable("cell(1,3)", DiscreteDomain((1, 2, 3, 4, 5, 6, 7, 8, 9)))
    assert fill.constraints[0] == Equals(StateVariable("payoff"), Constant(None))
    assert fill.constraints[1] == AllDifferent(
        (
            StateVariable("cell(1,1)"),
            StateVariable("cell(1,2)"),
            ActionParameter("cell(1,3)"),
            ActionParameter("cell(1,4)"),
            StateVariable("cell(1,5)"),
            ActionParameter("cell(1,6)"),
            ActionParameter("cell(1,7)"),
            ActionParameter("cell(1,8)"),
            ActionParameter("cell(1,9)"),
        )
    )


def test_transitions_write_every_parameter_into_its_cell_and_pay_one() -> None:
    (transition,) = create_sudoku_transitions().transitions
    (branch,) = transition.branches

    assert (transition.action, branch.probability, len(branch.effects)) == ("fill", 1.0, 52)
    assert branch.effects[0] == Assign(StateVariable("cell(1,3)"), ActionParameter("cell(1,3)"))
    assert branch.effects[-1] == Assign(StateVariable("payoff"), Constant(1.0))


def test_players_are_a_single_solver() -> None:
    assert create_sudoku_players() == Players(("solver",), "turn", ("payoff",))


def test_domain_holds_the_sudoku_recipes() -> None:
    assert create_sudoku_domain() == Domain(
        "sudoku",
        create_sudoku_initial_state(),
        create_sudoku_problem(),
        create_sudoku_transitions(),
        create_sudoku_players(),
    )


def test_a_name_and_a_grid_make_a_domain_of_that_puzzle() -> None:
    domain = create_sudoku_domain("sudoku/top95/1", TOP95_FIRST)
    variables = dict(domain.initial_state.variables)
    (fill,) = domain.problem.actions
    ((branch,),) = (transition.branches for transition in domain.transitions.transitions)

    assert domain.name == "sudoku/top95/1"
    assert (variables["cell(1,1)"], variables["cell(1,2)"], variables["cell(1,7)"]) == (4, None, 8)
    assert (len(fill.variables), len(branch.effects)) == (64, 65)


@pytest.mark.parametrize("grid", [TOP95_FIRST[:80], TOP95_FIRST + ".", TOP95_FIRST.replace(".", "0", 1)])
def test_a_grid_that_is_not_81_empty_marks_or_digits_raises(grid: str) -> None:
    with pytest.raises(ValueError, match="A sudoku grid needs 81 characters"):
        create_sudoku_initial_state(grid)
