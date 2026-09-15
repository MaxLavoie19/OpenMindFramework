import copy
import os
import pickle
import subprocess
import sys
from dataclasses import replace

from openmind.world.model.state import State

STATE = State((("cell(1,1)", "X"), ("turn", "O")))


def test_a_state_hashes_as_its_variables_and_equals_a_state_with_the_same_ones() -> None:
    same = State(STATE.variables)

    assert hash(STATE) == hash(STATE.variables) == hash(same)
    assert {STATE: 1}[same] == 1 and STATE == same and repr(STATE) == "State(variables=(('cell(1,1)', 'X'), ('turn', 'O')))"
    assert replace(STATE, variables=(("turn", "X"),)) == State((("turn", "X"),))
    assert copy.copy(STATE) == STATE and copy.deepcopy(STATE) == STATE


def test_a_copy_sent_to_a_process_that_hashes_names_another_way_hashes_its_own_way() -> None:
    hash(STATE)
    script = (
        "import pickle, sys\n"
        "from openmind.world.model.state import State\n"
        "state = pickle.loads(sys.stdin.buffer.read())\n"
        "print({State(state.variables): 1}.get(state, 0))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        input=pickle.dumps(STATE),
        capture_output=True,
        env={**os.environ, "PYTHONHASHSEED": "12345"},
        check=True,
    )

    assert result.stdout.strip() == b"1"
