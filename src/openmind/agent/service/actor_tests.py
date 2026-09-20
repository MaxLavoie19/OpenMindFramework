from collections.abc import Callable

from openmind.agent.factory.agent_factory import create_actor
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.strategy import Strategy
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.world import World

type Game = Callable[[str], RuleBasedGame]

HERE, THERE = State.of(turn="X"), State.of(turn="O")
PLACE = Action("place", (("col", 1), ("row", 1)))


class Dispatched:
    """A dispatcher keeping what it was asked to perform."""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def dispatch(self, action: Action) -> None:
        self.actions.append(action)

    def performing(self) -> tuple[Action, ...]:
        return ()


def test_the_actor_dispatches_what_the_strategy_says_to_play_here() -> None:
    dispatched = Dispatched()
    actor = create_actor(dispatched)
    actor.follow(Strategy.of(HERE, PLACE))

    assert actor.act(World(HERE))
    assert dispatched.actions == [PLACE]


def test_the_actor_waits_where_the_strategy_has_nothing_prepared_for_the_state() -> None:
    dispatched = Dispatched()
    actor = create_actor(dispatched)
    actor.follow(Strategy.of(HERE, PLACE))

    assert not actor.act(World(THERE))
    assert dispatched.actions == []


def test_the_actor_waits_until_it_is_given_a_strategy() -> None:
    dispatched = Dispatched()

    assert not create_actor(dispatched).act(World(HERE))


def test_a_state_already_acted_in_is_not_acted_in_twice() -> None:
    dispatched = Dispatched()
    actor = create_actor(dispatched)
    actor.follow(Strategy.of(HERE, PLACE))
    world = World(HERE)

    actor.act(world)
    actor.act(world)

    assert dispatched.actions == [PLACE]


def test_the_actor_acts_in_a_thread_of_its_own_until_it_is_stopped() -> None:
    dispatched = Dispatched()
    actor = create_actor(dispatched)
    world = World(HERE)
    actor.follow(Strategy.of(HERE, PLACE))

    actor.start(world)
    try:
        while not dispatched.actions:
            pass
    finally:
        actor.stop()

    assert dispatched.actions == [PLACE]
    assert not actor.acting()
