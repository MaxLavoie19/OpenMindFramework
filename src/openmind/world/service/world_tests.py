import threading

from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.world import World

HERE, THERE = State.of(turn="X"), State.of(turn="O")


def test_the_state_pushed_is_what_the_world_holds() -> None:
    world = World(HERE)

    assert world.current() == HERE
    assert world.perceived(THERE) == THERE
    assert world.current() == THERE


def test_what_happened_replaces_the_state_and_counts_as_a_change() -> None:
    world = World(HERE)

    world.happened(Action("place", ()), THERE)

    assert (world.current(), world.changes()) == (THERE, 1)


def test_perceiving_the_same_state_again_changes_nothing() -> None:
    world = World(HERE)

    world.perceived(HERE)

    assert world.changes() == 0


def test_the_state_is_read_and_replaced_from_several_threads() -> None:
    world = World(HERE)
    read: list[State] = []

    def perceive() -> None:
        for _ in range(200):
            world.perceived(THERE)
            world.perceived(HERE)

    def look() -> None:
        for _ in range(200):
            read.append(world.current())

    threads = [threading.Thread(target=perceive), threading.Thread(target=look)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert set(read) <= {HERE, THERE}
    assert world.current() == HERE
