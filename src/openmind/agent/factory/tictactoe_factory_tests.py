from openmind.agent.factory.tictactoe_factory import (
    create_tictactoe_domain,
    create_tictactoe_initial_state,
    create_tictactoe_problem,
    create_tictactoe_transitions,
)
from openmind.agent.model.domain import Domain
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable


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
                (
                    Equals(StateVariable("payoff", (Constant("X"),)), Constant(None)),
                    Equals(StateVariable("payoff", (Constant("O"),)), Constant(None)),
                    Equals(
                        StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))),
                        Constant(None),
                    ),
                ),
            ),
        )
    )


def test_transitions_give_place_one_certain_branch() -> None:
    (transition,) = create_tictactoe_transitions().transitions
    (branch,) = transition.branches

    assert (transition.action, branch.probability) == ("place", 1.0)


def test_domain_holds_the_tictactoe_recipes() -> None:
    assert create_tictactoe_domain() == Domain(
        "tictactoe",
        create_tictactoe_initial_state(),
        create_tictactoe_problem(),
        create_tictactoe_transitions(),
    )
