import pickle

from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.inference.constant.inference_constant import MEMORY_CHECK_INTERVAL
from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.service.consequence_library_tests import position, strip_domain
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State


def new_mechanics() -> Mechanics:
    return Mechanics(
        SolverBuilder().build(), PredictorBuilder().build(), StateNamespaceMapper(VariableNameMapper()), MemoryMeter()
    )


def test_the_same_state_gives_the_same_view() -> None:
    mechanics, domain = new_mechanics(), strip_domain()

    assert mechanics.view(domain, position({1: "X"}, "O")) is mechanics.view(domain, position({1: "X"}, "O"))


def test_moves_give_the_turn_to_the_player_and_lead_to_views_of_the_outcomes() -> None:
    mechanics, domain = new_mechanics(), strip_domain()

    moves = mechanics.moves(domain, position({1: "X"}, "O"), "X")

    assert [[(view.cell, probability) for view, probability in outcomes] for outcomes in moves] == [
        [({(1, 1): "X", (1, 2): "X", (1, 3): None, (1, 4): None}, 1.0)],
        [({(1, 1): "X", (1, 2): None, (1, 3): "X", (1, 4): None}, 1.0)],
        [({(1, 1): "X", (1, 2): None, (1, 3): None, (1, 4): "X"}, 1.0)],
    ]


def test_views_are_cleared_once_the_process_holds_more_than_its_share() -> None:
    mechanics, domain, state = new_mechanics(), strip_domain(), position({1: "X"}, "O")
    first = mechanics.view(domain, state)
    mechanics.limit_memory(1)

    for index in range(MEMORY_CHECK_INTERVAL):
        mechanics.view(domain, State((("clock", index),)))

    assert mechanics.view(domain, state) is not first


def test_clearing_forgets_every_view() -> None:
    mechanics, domain, state = new_mechanics(), strip_domain(), position({1: "X"}, "O")
    first = mechanics.view(domain, state)

    mechanics.clear()

    assert mechanics.view(domain, state) is not first


def test_a_copy_sent_to_another_process_leaves_its_views_behind_and_keeps_its_share() -> None:
    mechanics, domain, state = new_mechanics(), strip_domain(), position({1: "X"}, "O")
    mechanics.limit_memory(123)
    mechanics.moves(domain, state, "X")

    copy = pickle.loads(pickle.dumps(mechanics))

    assert copy.__getstate__()["_memory_share"] == 123
    assert len(copy.moves(domain, state, "X")) == 3
