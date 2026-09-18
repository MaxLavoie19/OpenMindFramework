import pickle
from dataclasses import replace

from openmind.inference.constant.inference_constant import MEMORY_CHECK_INTERVAL
from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.consequence_library_tests import Declare, position, strip_domain
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State


def new_mechanics() -> Mechanics:
    return Mechanics(StateNamespaceMapper(VariableNameMapper()), MemoryMeter())


def test_the_same_state_gives_the_same_view(declared: Declare) -> None:
    mechanics, rbs = new_mechanics(), strip_domain(declared)

    assert mechanics.view(rbs, position({1: "X"}, "O")) is mechanics.view(rbs, position({1: "X"}, "O"))


def test_moves_give_the_turn_to_the_player_and_lead_to_views_of_the_outcomes(declared: Declare) -> None:
    mechanics, rbs = new_mechanics(), strip_domain(declared)

    moves = mechanics.moves(rbs, position({1: "X"}, "O"), "X")

    assert [[(view.cell, probability) for view, probability in outcomes] for outcomes in moves] == [
        [({(1, 1): "X", (1, 2): "X", (1, 3): None, (1, 4): None}, 1.0)],
        [({(1, 1): "X", (1, 2): None, (1, 3): "X", (1, 4): None}, 1.0)],
        [({(1, 1): "X", (1, 2): None, (1, 3): None, (1, 4): "X"}, 1.0)],
    ]


def test_changes_count_the_actions_changing_each_indexed_variable(declared: Declare) -> None:
    mechanics, rbs, state = new_mechanics(), strip_domain(declared), position({1: "X"}, "O")

    changes = mechanics.changes(rbs, state, "X")

    # Marking the second cell also wins: both payoffs change once.
    assert changes == {("cell", (1, 2)): 1.0, ("cell", (1, 3)): 1.0, ("cell", (1, 4)): 1.0, ("payoff", "O"): 1.0, ("payoff", "X"): 1.0}
    view = mechanics.view(rbs, state)
    assert (view.changed("X", "cell", (1, 3)), view.changed("X", "cell", (1, 1)), view.changed("O", "cell", (1, 1))) == (1.0, 0.0, 0.0)


def test_changes_read_by_name_when_an_outcome_adds_variables(declared: Declare) -> None:
    effects = PythonRule("cell[1, col] = turn\ncell[2, col] = turn\nturn = 'O' if turn == 'X' else 'X'")
    rbs = declared(
        position({}, "X"),
        legal={"place": (PythonRule("payoff['X'] is None"), PythonRule("cell[1, col] is None"))},
        outcomes={"place": ((1.0, effects),)},
        players=Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)")),
        parameters={"place": {"col": PythonRule("(1, 2, 3, 4)")}},
        empties={"cell": None},
        context="wide strip",
    )

    changes = new_mechanics().changes(rbs, position({}, "X"), "X")

    assert changes == {("cell", (1, col)): 1.0 for col in range(1, 5)}


def test_views_are_cleared_once_the_process_holds_more_than_its_share(declared: Declare) -> None:
    mechanics, rbs, state = new_mechanics(), strip_domain(declared), position({1: "X"}, "O")
    first = mechanics.view(rbs, state)
    mechanics.limit_memory(1)

    for index in range(MEMORY_CHECK_INTERVAL):
        mechanics.view(rbs, State((("clock", index),)))

    assert mechanics.view(rbs, state) is not first


def test_clearing_forgets_every_view(declared: Declare) -> None:
    mechanics, rbs, state = new_mechanics(), strip_domain(declared), position({1: "X"}, "O")
    first = mechanics.view(rbs, state)

    mechanics.clear()

    assert mechanics.view(rbs, state) is not first


def test_a_copy_sent_to_another_process_leaves_its_views_behind_and_keeps_its_share(declared: Declare) -> None:
    mechanics, rbs, state = new_mechanics(), strip_domain(declared), position({1: "X"}, "O")
    mechanics.limit_memory(123)
    mechanics.moves(rbs, state, "X")

    copy = pickle.loads(pickle.dumps(mechanics))

    assert copy.__getstate__()["_memory_share"] == 123
    assert len(copy.moves(rbs, state, "X")) == 3
