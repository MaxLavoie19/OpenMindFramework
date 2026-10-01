import pickle

from openmind.heuristic.model.node import Node
from openmind.world.model.state import State

START = State.of(turn="X")


def test_a_feature_is_extracted_the_first_time_it_is_asked_for_and_shared_after() -> None:
    node = Node(START)
    extracted = []

    def extract() -> object:
        extracted.append("mobility")
        return 3

    assert (node.feature("mobility", extract), node.feature("mobility", extract)) == (3, 3)
    assert extracted == ["mobility"]


def test_a_node_of_another_state_keeps_the_game_and_extracts_its_own_features() -> None:
    node = Node(START, game="the game")
    node.feature("mobility", lambda: 3)

    after = node.of(State.of(turn="O"))

    assert (after.game, after.state.value("turn"), after.features) == ("the game", "O", {})


def test_a_node_whose_features_hold_functions_can_still_be_sent_to_another_process() -> None:
    """**A node is read in one process and judged in another, and a feature can hold a function.**
    `RuleHeuristic` keeps the consequence library's names as a feature, and `win chance`, `wins` and `near` are
    closures over that process's library. Pickled with them a node raises, so a node that had been valued could
    not be sent — which is how a judging died after the teller valued its decisions here and the payoff judging
    sent the same decisions to workers."""
    node = Node(START, game="the game")
    node.feature("consequences of X", lambda: {"win chance": lambda action: 0.5})

    after = pickle.loads(pickle.dumps(node))

    assert (after.state.value("turn"), after.game) == ("X", "the game")


def test_a_node_sent_to_another_process_arrives_ready_to_extract_its_own_features() -> None:
    """The work stays behind, not the ability to do it. A worker that receives a bare node extracts what it
    needs, which is what makes leaving the features behind cost time rather than correctness."""
    node = Node(START)
    node.feature("mobility", lambda: 3)

    after = pickle.loads(pickle.dumps(node))

    # Copied before the call, because `features` is the live dict and the call below fills it.
    arrived = dict(after.features)

    assert (arrived, after.feature("mobility", lambda: 7)) == ({}, 7)
