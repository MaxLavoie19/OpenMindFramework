import pickle

import pytest

from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.grid import Grid
from openmind.world.model.state import State

STATE = State((("cell(1,1)", "X"), ("cell(1,2)", None), ("payoff(O)", None), ("payoff(X)", None), ("turn", "O")))


def new_mapper() -> StateNamespaceMapper:
    return StateNamespaceMapper(VariableNameMapper())


def test_plain_variables_are_values_and_indexed_ones_are_gathered_under_their_base() -> None:
    assert new_mapper().to_namespace(STATE) == {
        "cell": {(1, 1): "X", (1, 2): None},
        "payoff": {"O": None, "X": None},
        "turn": "O",
    }


def test_a_base_indexed_by_whole_numbers_is_a_grid_and_rules_still_write_it() -> None:
    mapper = new_mapper()
    state = State((*STATE.variables[:2], ("lamp(1)", True), ("lamp(2)", False), *STATE.variables[2:]))

    namespace = mapper.to_namespace(state)
    namespace["cell"][1, 2] = "O"  # type: ignore[index]

    assert isinstance(namespace["cell"], Grid) and isinstance(namespace["lamp"], Grid)
    assert not isinstance(namespace["payoff"], Grid)
    assert (namespace["cell"].where("O"), namespace["lamp"].where(True)) == (((1, 2),), (1,))  # type: ignore[attr-defined]
    assert dict(mapper.to_state(state, namespace).variables)["cell(1,2)"] == "O"


def test_to_state_reads_the_changed_values_back() -> None:
    mapper = new_mapper()
    namespace = mapper.to_namespace(STATE)
    namespace["cell"][1, 2] = "O"  # type: ignore[index]
    namespace["turn"] = "X"
    namespace["row"] = 1

    assert mapper.to_state(STATE, namespace) == State(
        (("cell(1,1)", "X"), ("cell(1,2)", "O"), ("payoff(O)", None), ("payoff(X)", None), ("turn", "X"))
    )


def test_an_index_added_under_a_base_the_state_has_is_a_new_variable_after_its_own() -> None:
    mapper = new_mapper()
    namespace = mapper.to_namespace(STATE)
    namespace["payoff"]["Z"] = 1.0  # type: ignore[index]
    namespace["cell"][2, 1] = "O"  # type: ignore[index]

    assert mapper.to_state(STATE, namespace) == State((*STATE.variables, ("cell(2,1)", "O"), ("payoff(Z)", 1.0)))


@pytest.mark.parametrize("index", ["1", "a,b", True, 1.5])
def test_an_added_index_that_a_variable_name_can_t_read_back_raises(index: object) -> None:
    mapper = new_mapper()
    namespace = mapper.to_namespace(STATE)
    namespace["payoff"][index] = 1.0  # type: ignore[index]

    with pytest.raises(ValueError, match="can't be written in a variable name"):
        mapper.to_state(STATE, namespace)


def test_to_source_writes_how_a_rule_reads_a_variable() -> None:
    mapper = new_mapper()

    assert [mapper.to_source(name) for name in ("turn", "cell(2,3)", "payoff(X)", "lamp(1)")] == [
        "turn",
        "cell[2, 3]",
        "payoff['X']",
        "lamp[1]",
    ]


def test_a_copy_sent_to_another_process_lays_states_out_again() -> None:
    mapper = new_mapper()
    state = State((("cell(1,1)", "X"), ("turn", "O")))
    mapper.to_namespace(state)

    copy = pickle.loads(pickle.dumps(mapper))

    assert copy.to_namespace(state) == {"cell": {(1, 1): "X"}, "turn": "O"}


def test_a_base_used_both_plain_and_indexed_raises() -> None:
    with pytest.raises(ValueError, match="lamp"):
        new_mapper().to_namespace(State((("lamp", 1), ("lamp(1)", 2))))
