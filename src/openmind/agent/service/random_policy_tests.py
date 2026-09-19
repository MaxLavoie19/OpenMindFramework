from collections.abc import Callable

import random

import pytest

from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.agent.service.random_policy import RandomPolicy
from openmind.timing.model.clock import Clock
from openmind.structure.model.map import Map

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]


def test_chooses_a_legal_action(game: Game) -> None:
    rbs = game("tictactoe")

    action = RandomPolicy(random.Random(1)).choose(rbs, rbs.start())

    assert action in rbs.actions(rbs.start())


def test_the_same_seed_gives_the_same_choices(game: Game) -> None:
    rbs = game("tictactoe")
    first, second = RandomPolicy(random.Random(7)), RandomPolicy(random.Random(7))

    assert [first.choose(rbs, rbs.start()) for _ in range(5)] == [
        second.choose(rbs, rbs.start()) for _ in range(5)
    ]


def test_no_legal_action_raises(game: Game) -> None:
    rbs = game("tictactoe")
    finished = rbs.start().with_model("payoff", Map.of({"X": 1.0, "O": 0.0}))

    with pytest.raises(ValueError, match="No legal action"):
        RandomPolicy(random.Random(1)).choose(rbs, finished)


def test_a_clock_changes_nothing(game: Game) -> None:
    rbs = game("tictactoe")
    first, second = RandomPolicy(random.Random(7)), RandomPolicy(random.Random(7))

    assert first.choose(rbs, rbs.start()) == second.choose(rbs, rbs.start(), clock=Clock(1.0), steps_played=4)
