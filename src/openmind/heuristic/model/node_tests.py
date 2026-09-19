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
