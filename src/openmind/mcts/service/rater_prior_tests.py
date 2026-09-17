import math

import pytest

from openmind.mcts.service.rater_prior import RaterPrior
from openmind.mcts.service.uniform_prior_tests import opening
from openmind.world.model.action import Action
from openmind.world.model.state import State

CENTER = Action("place", (("col", 2), ("row", 2)))


class FavourCenter:
    """Rates the center 1.0, the corners nothing, and every other cell 0.0."""

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        corners = {(1, 1), (1, 3), (3, 1), (3, 3)}
        cells = [(dict(action.parameters)["row"], dict(action.parameters)["col"]) for action in actions]
        return tuple(1.0 if action == CENTER else None if cell in corners else 0.0 for action, cell in zip(actions, cells))


def test_the_rater_prior_follows_the_ratings_and_gives_an_unrated_action_the_mean() -> None:
    state, actions = opening()

    priors = dict(zip(actions, RaterPrior(FavourCenter(), 0.1).priors(state, actions), strict=True))

    assert math.isclose(sum(priors.values()), 1.0)
    edge, corner = Action("place", (("col", 2), ("row", 1))), Action("place", (("col", 1), ("row", 1)))
    assert priors[edge] < priors[corner] < priors[CENTER]


def test_a_temperature_of_zero_raises() -> None:
    with pytest.raises(ValueError, match="above 0"):
        RaterPrior(FavourCenter(), 0.0)
