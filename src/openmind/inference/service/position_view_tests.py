import pytest

from openmind.inference.constant.inference_constant import OUTSIDE
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


def test_rules_read_the_view_as_here() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "O")
    rule = RuleCompiler().compile_value(PythonRule("here.best(other, lambda v1: v1.payoff[other] == 1.0)"))

    assert RuleRunner(StateNamespaceMapper(VariableNameMapper())).value(rule, state, None, library.names(domain, state)) == 1.0
