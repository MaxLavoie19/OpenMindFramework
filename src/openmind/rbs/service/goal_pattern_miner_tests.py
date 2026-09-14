from openmind.csp.factory.csp_factory import create_solver
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.service.consequence_library_tests import place, position, strip_domain
from openmind.rbs.service.goal_pattern_miner import GoalPatternMiner
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


def new_miner() -> GoalPatternMiner:
    names = VariableNameMapper()
    return GoalPatternMiner(ConsequenceLibraryBuilder().build(), create_solver(), StateNamespaceMapper(names), names)


def test_a_winning_move_gives_the_variables_it_needs_relative_to_the_mover_and_the_move() -> None:
    rows = [
        ActionRow(position({1: "X", 4: "O"}, "X"), place(2), 10, 1.0, 0.0),
        ActionRow(position({1: "X"}, "O"), place(3), 10, 0.2, 0.0),
        ActionRow(position({3: "O", 1: "X"}, "O"), place(4), 10, 1.0, 0.0),
    ]

    assert new_miner().patterns(strip_domain(), rows, limit=10) == (
        GoalPattern("place", (PythonRule("near(action, 0, -1) == me"),), 2),
    )


def test_the_limit_bounds_how_many_winning_moves_are_probed() -> None:
    rows = [
        ActionRow(position({1: "X", 4: "O"}, "X"), place(2), 10, 1.0, 0.0),
        ActionRow(position({3: "O", 1: "X"}, "O"), place(4), 10, 1.0, 0.0),
    ]

    assert new_miner().patterns(strip_domain(), rows, limit=1)[0].moves == 1
