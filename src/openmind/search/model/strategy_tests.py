from openmind.search.model.strategy import Strategy
from openmind.world.model.action import Action
from openmind.world.model.state import State

HERE, THERE = State.of(turn="X"), State.of(turn="O")
LEFT, RIGHT = Action("go", (("way", "left"),)), Action("go", (("way", "right"),))


def test_a_mixed_strategy_holds_a_move_distribution_per_state() -> None:
    strategy = Strategy(((HERE, ((LEFT, 0.7), (RIGHT, 0.3))), (THERE, ((RIGHT, 1.0),))))

    assert strategy.at(HERE) == ((LEFT, 0.7), (RIGHT, 0.3))
    assert strategy.chosen(HERE) == LEFT
    assert strategy.chosen(THERE) == RIGHT
    assert strategy.states() == (HERE, THERE)


def test_a_state_the_strategy_says_nothing_about_is_where_to_strategize_again() -> None:
    strategy = Strategy.of(HERE, LEFT)

    assert strategy.at(THERE) == ()
    assert strategy.chosen(THERE) is None
