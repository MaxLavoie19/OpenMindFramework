import pickle

import pytest

from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
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


def test_to_state_reads_the_changed_values_back() -> None:
    mapper = new_mapper()
    namespace = mapper.to_namespace(STATE)
    namespace["cell"][1, 2] = "O"  # type: ignore[index]
    namespace["turn"] = "X"
    namespace["row"] = 1

    assert mapper.to_state(STATE, namespace) == State(
        (("cell(1,1)", "X"), ("cell(1,2)", "O"), ("payoff(O)", None), ("payoff(X)", None), ("turn", "X"))
    )


def test_to_state_rejects_an_index_the_state_does_not_have() -> None:
    mapper = new_mapper()
    namespace = mapper.to_namespace(STATE)
    namespace["payoff"]["Z"] = 1.0  # type: ignore[index]

    with pytest.raises(KeyError, match=r"payoff\(Z\)"):
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
