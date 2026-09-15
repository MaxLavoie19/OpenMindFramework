import pytest

from openmind.agent.model.domain import Domain
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.inference.constant.inference_constant import OUTSIDE
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.world.model.players import Players
from openmind.inference.service.mechanics_tests import new_mechanics
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.service.consequence_library_tests import coin_domain, position, strip_domain
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State


def test_variables_are_attributes_and_offsets_read_around_an_index() -> None:
    view = new_mechanics().view(strip_domain(), position({1: "X", 2: "O"}, "X"))

    assert (view.turn, view.cell[1, 2]) == ("X", "O")
    assert [view.offset("cell", (1, 1), *steps) for steps in ((0, 1), (0, 3), (0, -1), (1, 0), (0,))] == [
        "O",
        None,
        OUTSIDE,
        OUTSIDE,
        OUTSIDE,
    ]
    with pytest.raises(AttributeError, match="no variable 'board'"):
        _ = view.board


def test_best_worst_and_count_look_one_action_ahead_for_either_player() -> None:
    view = new_mechanics().view(strip_domain(), position({1: "X"}, "O"))

    def wins(player: str):  # type: ignore[no-untyped-def]
        return lambda after: after.payoff[player] == 1.0

    assert (view.best("X", wins("X")), view.worst("X", wins("X")), view.count("X", wins("X"))) == (1.0, 0.0, 1.0)
    assert (view.best("O", wins("O")), view.count("O", wins("O"))) == (0.0, 0.0)
    assert view.mobility("X") == 3


def test_look_aheads_nest_and_read_the_views_they_close_over() -> None:
    view = new_mechanics().view(strip_domain(), position({1: "X"}, "O"))

    guaranteed =view.worst("O", lambda reply: reply.best("X", lambda move: move.payoff["X"] == 1.0))
    marks_gained = view.best("X", lambda move: sum(mark == "X" for mark in move.cell.values()) - sum(mark == "X" for mark in view.cell.values()))

    assert (guaranteed, marks_gained) == (0.0, 1.0)


def test_a_player_without_an_action_leaves_the_reading_as_it_is_here() -> None:
    finished = dict(position({1: "X", 2: "X"}, "O").variables)
    view = new_mechanics().view(strip_domain(), State(tuple(sorted({**finished, "payoff(X)": 1.0, "payoff(O)": 0.0}.items()))))

    assert (view.best("O", lambda after: after.payoff["X"]), view.count("O", lambda after: True)) == (1.0, 0.0)


def test_an_action_s_outcomes_are_weighted_by_their_probability() -> None:
    domain = coin_domain()

    assert new_mechanics().view(domain, domain.initial_state).best("A", lambda after: after.payoff["A"] == 1.0) == 0.5


def capture_domain() -> Domain:
    """A made-up game: A and B each have pieces on a row of 7 cells; the player to act moves one of their pieces one cell
    left or right onto an empty cell or onto the other player's piece, taking it; taking the last one wins."""
    effects = PythonRule(
        "cell[at + step] = turn\n"
        "cell[at] = None\n"
        "if ('B' if turn == 'A' else 'A') not in cell.values():\n"
        "    payoff[turn], payoff['B' if turn == 'A' else 'A'] = 1.0, 0.0\n"
        "turn = 'B' if turn == 'A' else 'A'"
    )
    constraints = (
        PythonRule("payoff['A'] is None"),
        PythonRule("payoff['B'] is None"),
        PythonRule("cell[at] == turn"),
        PythonRule("1 <= at + step <= 7 and cell[at + step] != turn"),
    )
    variables = (Variable("at", DiscreteDomain(tuple(range(1, 8)))), Variable("step", DiscreteDomain((-1, 1))))
    return Domain(
        "capture",
        capture_position({1: "A", 3: "A", 5: "B", 7: "B"}, "A"),
        Problem((ActionDefinition("move", variables, constraints),)),
        TransitionModel((Transition("move", (Branch(1.0, effects),)),)),
        Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
    )


def capture_position(pieces: dict[int, str], turn: str) -> State:
    variables = {**{f"cell({at})": pieces.get(at) for at in range(1, 8)}, "payoff(A)": None, "payoff(B)": None, "turn": turn}
    return State(tuple(sorted(variables.items())))


def test_what_if_views_edit_a_copy_of_the_grids_and_leave_the_position_as_it_is() -> None:
    mechanics, domain = new_mechanics(), strip_domain()
    view = mechanics.view(domain, position({1: "X", 2: "O"}, "X"))

    assert mechanics.empties(domain) == {"cell": None}
    assert view.with_value("cell", (1, 4), "O").cell == {(1, 1): "X", (1, 2): "O", (1, 3): None, (1, 4): "O"}
    assert view.cleared((1, 1)).cell == {(1, 1): None, (1, 2): "O", (1, 3): None, (1, 4): None}
    assert view.copied((1, 1), (1, 3)).cell == {(1, 1): "X", (1, 2): "O", (1, 3): "X", (1, 4): None}
    assert view.alone((1, 2)).cell == {(1, 1): None, (1, 2): "O", (1, 3): None, (1, 4): None}
    assert (view.cell[1, 4], view.alone((1, 2)).turn) == (None, "X")
    with pytest.raises(KeyError, match="no variable"):
        view.with_value("cell", (9, 9), "X")


def test_what_if_views_read_defenders_lone_moves_and_attacks_on_empty_cells() -> None:
    mechanics, domain = new_mechanics(), capture_domain()
    view = mechanics.view(domain, capture_position({2: "A", 3: "A", 5: "B"}, "A"))

    # If A's piece on 3 were B's, A's piece on 2 could take it; B's piece on 5 has nobody next to it.
    defended = (view.with_value("cell", 3, "B").changed("A", "cell", 3), view.with_value("cell", 5, "A").changed("B", "cell", 5))
    # Alone, a piece on 3 has two moves and one on 7 only one; B's piece copied onto the empty 4 could be taken by A's 3.
    assert defended == (1.0, 0.0)
    assert (view.alone(3).mobility("A"), view.copied(3, 7).alone(7).mobility("A")) == (2, 1)
    assert view.copied(5, 4).changed("A", "cell", 4) == 1.0


def test_rules_read_the_view_as_here() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "O")
    rule = RuleCompiler().compile_value(PythonRule("here.best(other, lambda v1: v1.payoff[other] == 1.0)"))

    assert RuleRunner(StateNamespaceMapper(VariableNameMapper())).value(rule, state, None, library.names(domain, state)) == 1.0
