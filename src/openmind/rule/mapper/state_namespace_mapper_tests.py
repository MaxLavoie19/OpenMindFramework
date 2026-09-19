import pytest

from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.map import Map
from openmind.world.model.state import State

BOARD = Grid.filled((2, 2), None)
STATE = State.of(cell=BOARD, history=List(()), payoff=Map.of({"X": None}), turn="X")


def test_a_scalar_reads_as_its_value_and_every_other_model_as_itself() -> None:
    namespace = StateNamespaceMapper().to_namespace(STATE)

    assert namespace == {"cell": BOARD, "history": List(()), "payoff": Map.of({"X": None}), "turn": "X"}


def test_a_script_s_assignments_are_read_back_as_models() -> None:
    namespace = {**StateNamespaceMapper().to_namespace(STATE), "turn": "O", "cell": BOARD.placed((1, 1), "X")}

    assert StateNamespaceMapper().to_state(STATE, namespace) == STATE.with_model("turn", "O").with_model(
        "cell", BOARD.placed((1, 1), "X")
    )


def test_a_scalar_written_a_model_or_a_model_written_something_else_is_refused() -> None:
    namespace = StateNamespaceMapper().to_namespace(STATE)

    with pytest.raises(ValueError, match="turn is a Scalar"):
        StateNamespaceMapper().to_state(STATE, {**namespace, "turn": BOARD})
    with pytest.raises(ValueError, match="payoff is a Map"):
        StateNamespaceMapper().to_state(STATE, {**namespace, "payoff": {"X": 1.0}})
