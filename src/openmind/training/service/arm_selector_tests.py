import math
import random

import pytest

from openmind.training.service.arm_selector import ArmSelector

EXPLORATION = math.sqrt(2)


def test_arms_never_chosen_go_first() -> None:
    first, second = ArmSelector().pair({"win": (10, 6.0)}, {}, ("win", "pieces", "mobility"), EXPLORATION, random.Random(1))

    assert {first, second} == {"pieces", "mobility"}


def test_between_arms_played_as_often_the_better_score_is_chosen_first() -> None:
    scores = {"win": (20, 16.0), "pieces": (20, 8.0), "mobility": (20, 4.0)}

    assert ArmSelector().pair(scores, {}, ("mobility", "pieces", "win"), EXPLORATION, random.Random(1)) == ("win", "pieces")


def test_games_under_way_make_an_arm_less_likely_to_be_chosen_again() -> None:
    scores = {"win": (20, 12.0), "pieces": (20, 12.0), "mobility": (20, 12.0)}

    first, second = ArmSelector().pair(scores, {"win": 30}, ("win", "pieces", "mobility"), EXPLORATION, random.Random(1))

    assert {first, second} == {"pieces", "mobility"}


def test_a_game_needs_two_arms() -> None:
    with pytest.raises(ValueError, match="two arms, not 1"):
        ArmSelector().pair({}, {}, ("win", "win"), EXPLORATION, random.Random(1))
