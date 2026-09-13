from openmind.agent.factory.tictactoe_factory import (
    create_tictactoe_initial_state,
    create_tictactoe_problem,
    create_tictactoe_transitions,
)
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value


def play(*cells: tuple[int, int]) -> State:
    names = VariableNameMapper()
    predictor = Predictor(Interpreter(names), names, ExpressionTextMapper(names), ActionTextMapper())
    transitions = create_tictactoe_transitions()
    state = create_tictactoe_initial_state()
    for row, col in cells:
        distribution = predictor.predict(transitions, state, Action("place", (("col", col), ("row", row))))
        ((state, probability),) = distribution.outcomes
        assert probability == 1.0
    return state


def values_of(state: State, *names: str) -> tuple[Value, ...]:
    variables = dict(state.variables)
    return tuple(variables[name] for name in names)


def legal_action_count(state: State) -> int:
    names = VariableNameMapper()
    solver = Solver(Interpreter(names), ExpressionTextMapper(names), ActionTextMapper())
    return len(solver.solve(create_tictactoe_problem(), state))


def test_turns_alternate() -> None:
    assert values_of(play((2, 2)), "cell(2,2)", "turn") == ("X", "O")
    assert values_of(play((2, 2), (1, 1)), "cell(2,2)", "cell(1,1)", "turn") == ("X", "O", "X")


def test_winning_move_sets_payoffs_and_ends_the_game() -> None:
    # X takes row 1 while O takes (2,1) and (2,2)
    state = play((1, 1), (2, 1), (1, 2), (2, 2), (1, 3))

    assert values_of(state, "payoff(X)", "payoff(O)") == (1.0, 0.0)
    assert legal_action_count(state) == 0


def test_full_board_without_a_line_is_a_draw() -> None:
    # X O X / X O O / O X X
    state = play((1, 1), (1, 2), (1, 3), (2, 2), (2, 1), (2, 3), (3, 2), (3, 1), (3, 3))

    assert values_of(state, "payoff(X)", "payoff(O)") == (0.5, 0.5)
    assert legal_action_count(state) == 0


def test_line_completed_with_the_last_empty_cell_is_a_win() -> None:
    # X X O / O X X / O O X, X completes the diagonal with (3,3)
    state = play((1, 1), (1, 3), (1, 2), (2, 1), (2, 2), (3, 1), (2, 3), (3, 2), (3, 3))

    assert values_of(state, "payoff(X)", "payoff(O)") == (1.0, 0.0)
