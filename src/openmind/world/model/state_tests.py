import copy
import os
import pickle
import subprocess
import sys
from dataclasses import replace

from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.world.model.state import State

STATE = State.of(cell=Grid.filled((3, 3), None).placed((1, 1), "X"), turn="O", payoff=Map.of({"X": None, "O": None}))


def test_a_state_is_named_data_models_sorted_by_name_a_plain_value_being_a_scalar() -> None:
    assert STATE.names() == ("cell", "payoff", "turn")
    assert STATE.model("turn") == Scalar("O") and STATE.value("turn") == "O"
    assert STATE.model("cell")[1, 1] == "X"  # type: ignore[index]


def test_a_state_hashes_as_its_models_and_equals_a_state_with_the_same_ones() -> None:
    same = State(STATE.models)

    assert hash(STATE) == hash(STATE.models) == hash(same)
    assert {STATE: 1}[same] == 1 and STATE == same
    assert replace(STATE, models=(("turn", Scalar("X")),)) == State.of(turn="X")
    assert copy.copy(STATE) == STATE and copy.deepcopy(STATE) == STATE


def test_setting_a_model_gives_a_new_state() -> None:
    moved = STATE.with_model("turn", "X")

    assert (moved.value("turn"), STATE.value("turn")) == ("X", "O")
    assert STATE.with_model("score", 3).names() == ("cell", "payoff", "score", "turn")


def test_a_copy_sent_to_a_process_that_hashes_names_another_way_hashes_its_own_way() -> None:
    hash(STATE)
    script = (
        "import pickle, sys\n"
        "from openmind.world.model.state import State\n"
        "state = pickle.loads(sys.stdin.buffer.read())\n"
        "print({State(state.models): 1}.get(state, 0))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        input=pickle.dumps(STATE),
        capture_output=True,
        env={**os.environ, "PYTHONHASHSEED": "12345"},
        check=True,
    )

    assert result.stdout.strip() == b"1"
