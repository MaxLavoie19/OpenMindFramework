from pathlib import Path
from collections.abc import Callable
import logging
import random
import re

import pytest

from openmind.agent.service.random_policy import RandomPolicy
from openmind.agent.service.timekeeper import Timekeeper
from openmind.agent.service.timekeeper_tests import TIMEOUT, alternating_steps
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.evaluation.model.match_game import MatchGame
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.model.python_rule import PythonRule
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class Always:
    """A policy that always chooses the same action."""

    def __init__(self, name: str) -> None:
        self._action = Action(name, ())

    def choose(self, rbs: RuleBasedSystem, state: State) -> Action:
        return self._action


type Game = Callable[[str], RuleBasedSystem]


type Declare = Callable[..., RuleBasedSystem]


def new_match_runner() -> MatchRunner:
    return MatchRunner(StateReader(), TaskRunner(1))


def timed(knowledge: KnowledgeBase, game: Game) -> RuleBasedSystem:
    """Rock paper scissors with a timeout rule declared into its context, so it can be played on a clock."""
    rbs = game("rockpaperscissors")
    RuleDeclarer(knowledge, rbs.context).timeout(TIMEOUT)
    return create_rule_based_system(knowledge, rbs.context)


def first_mover_decides(declared: Declare, players: tuple[str, ...] = ("A", "B")) -> RuleBasedSystem:
    """Only the first player acts: win gives them 1 and the other 0, lose the reverse, tie 0.5 each."""
    first, *others = players
    unset = PythonRule(f"payoff[{first!r}] is None")

    def payoffs(mine: float, theirs: float) -> PythonRule:
        return PythonRule(
            "\n".join((f"payoff[{first!r}] = {mine!r}", *(f"payoff[{other!r}] = {theirs!r}" for other in others)))
        )

    return declared(
        State((*((f"payoff({player})", None) for player in sorted(players)), ("turn", first))),
        legal={name: (unset,) for name in ("win", "lose", "tie")},
        outcomes={
            "win": ((1.0, payoffs(1.0, 0.0)),),
            "lose": ((1.0, payoffs(0.0, 1.0)),),
            "tie": ((1.0, payoffs(0.5, 0.5)),),
        },
        players=Players(players, "turn", tuple(f"payoff({player})" for player in players)),
        context="first mover decides",
    )


def series(declared: Declare, evaluated: str, opponent: str, games: int = 2) -> MatchResults:
    return new_match_runner().series(
        first_mover_decides(declared),
        lambda seed: Always(evaluated),
        lambda seed: Always(opponent),
        "always " + opponent,
        games,
        random.Random(1),
    )


def test_each_game_is_logged_as_it_ends(declared: Declare, game: Game, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation.service.match_runner")

    series(declared, "win", "lose")

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == "openmind.evaluation.service.match_runner" and record.levelno == logging.INFO
    ]
    assert [re.sub(r"seeds \d+ and \d+", "seeds", line) for line in lines] == [
        "Game with seeds finished in 1 plies, the evaluated policy playing A: payoffs A=1.0 B=0.0",
        "Game with seeds finished in 1 plies, the evaluated policy playing B: payoffs A=0.0 B=1.0",
    ]


def test_players_acting_at_once_each_choose_and_their_actions_are_taken_together(declared: Declare, game: Game, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation.service.match_runner")
    rbs = game("rockpaperscissors")

    def random_policy(seed: int) -> RandomPolicy:
        return RandomPolicy(random.Random(seed))

    results = new_match_runner().series(rbs, random_policy, random_policy, "random", 4, random.Random(1))

    lines = [record.getMessage() for record in caplog.records if record.name == "openmind.evaluation.service.match_runner"]
    assert results.games == 4 and results.wins + results.draws + results.losses == 4
    assert len([line for line in lines if "finished in 1 plies" in line]) == 4


def test_seats_switch_every_game(declared: Declare, game: Game) -> None:
    # game 1: the evaluated policy is A and wins; game 2: the opponent is A, loses, so the evaluated B wins
    assert series(declared, "win", "lose") == MatchResults("always lose", 2, 2, 0, 0)


def test_losses_and_wins_count_from_the_evaluated_side(declared: Declare, game: Game) -> None:
    assert series(declared, "lose", "lose") == MatchResults("always lose", 2, 1, 0, 1)


def test_equal_payoffs_are_draws(declared: Declare, game: Game) -> None:
    assert series(declared, "tie", "tie", games=3) == MatchResults("always tie", 3, 0, 3, 0)


def test_each_game_creates_its_policies_from_its_own_seed(declared: Declare, game: Game) -> None:
    seeds: list[int] = []

    def remember(seed: int) -> Always:
        seeds.append(seed)
        return Always("tie")

    new_match_runner().series(first_mover_decides(declared), remember, lambda seed: Always("tie"), "always tie", 3, random.Random(1))

    assert len(set(seeds)) == 3


class Recording:
    """A policy that always chooses the same action and records every state it is given."""

    def __init__(self, name: str) -> None:
        self._action = Action(name, ())
        self.states: list[State] = []

    def choose(self, rbs: RuleBasedSystem, state: State) -> Action:
        self.states.append(state)
        return self._action


def test_a_policy_is_given_the_position_as_it_is(declared: Declare) -> None:
    rbs = first_mover_decides(declared)
    recording = Recording("win")

    new_match_runner().play_game(rbs, lambda seed: recording, lambda seed: Always("win"), 0, 1, 1)

    assert recording.states == [State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A")))]


def test_a_game_without_two_players_raises(declared: Declare, game: Game) -> None:
    with pytest.raises(ValueError, match="two players"):
        new_match_runner().series(
            first_mover_decides(declared, ("A",)), lambda seed: Always("win"), lambda seed: Always("win"), "always win", 1, random.Random(1)
        )


class Stepping:
    """A policy stepping every time, remembering the steps it's told it has played and the clocks it's given."""

    def __init__(self) -> None:
        self.told: list[tuple[int, Clock | None]] = []

    def choose(
        self, rbs: RuleBasedSystem, state: State, player: str | None = None, clock: Clock | None = None, steps_played: int = 0
    ) -> Action:
        self.told.append((steps_played, clock))
        return Action("step", ())


def clocked_match_runner() -> MatchRunner:
    """A match runner whose referee reads a time source moving on by a second a reading: every choice takes a second."""
    return MatchRunner(
        StateReader(), TaskRunner(1), timekeeper=Timekeeper(create_rule_caller(), Ticking())
    )


def test_on_a_clock_each_policy_is_given_its_clock_and_its_own_steps_played(declared: Declare, game: Game, tmp_path: Path) -> None:
    first, second = Stepping(), Stepping()

    game = clocked_match_runner().play_game(
        alternating_steps(TIMEOUT, tmp_path), lambda seed: first, lambda seed: second, 0, 1, 1, TimeControl(1.5, 1.0)
    )

    payoffs, flagged = game.payoffs, game.flagged
    assert payoffs == (0.5, 0.5) and flagged is None
    assert first.told == [(0, Clock(1.5, 1.0)), (1, Clock(1.5, 1.0))]
    assert second.told == [(0, Clock(1.5, 1.0)), (1, Clock(1.5, 1.0))]


def test_wins_and_losses_on_time_count_from_the_evaluated_side(declared: Declare, game: Game, tmp_path: Path) -> None:
    results = clocked_match_runner().series(
        alternating_steps(TIMEOUT, tmp_path), lambda seed: Stepping(), lambda seed: Stepping(), "stepping", 2, random.Random(1), TimeControl(1.5)
    )

    # A runs out of time on its second move: game 1 the evaluated policy is A and loses, game 2 it is B and wins
    assert results == MatchResults("stepping", 2, 1, 0, 1, TimeControl(1.5), 1, 1)


def test_players_acting_at_once_are_each_charged_on_their_own_clock(knowledge: KnowledgeBase, declared: Declare, game: Game, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation.service.match_runner")
    rbs = timed(knowledge, game)

    def random_policy(seed: int) -> RandomPolicy:
        return RandomPolicy(random.Random(seed))

    game = clocked_match_runner().play_game(rbs, random_policy, random_policy, 0, 1, 1, TimeControl(1.5))

    assert game.flagged is None
    assert any("finished in 1 plies on 0.025+0, clocks A=0.50 B=0.50, the evaluated policy" in message for message in caplog.messages)


def test_players_acting_at_once_whose_time_runs_out_end_the_game_by_the_timeout_rule(knowledge: KnowledgeBase, declared: Declare, game: Game) -> None:
    rbs = timed(knowledge, game)

    def random_policy(seed: int) -> RandomPolicy:
        return RandomPolicy(random.Random(seed))

    game = clocked_match_runner().play_game(rbs, random_policy, random_policy, 0, 1, 1, TimeControl(0.5))

    assert game.flagged == "A"


def test_a_series_on_a_clock_needs_a_timeout_rule(declared: Declare, game: Game) -> None:
    with pytest.raises(ValueError, match="no timeout rule"):
        clocked_match_runner().series(
            first_mover_decides(declared), lambda seed: Always("win"), lambda seed: Always("win"), "always win", 1, random.Random(1), TimeControl(60.0)
        )


def test_each_game_is_given_as_it_ends_with_the_evaluated_policy_s_seat(declared: Declare, game: Game) -> None:
    given: list[tuple[int, int, MatchGame]] = []

    new_match_runner().series(
        first_mover_decides(declared),
        lambda seed: Always("win"),
        lambda seed: Always("lose"),
        "always lose",
        3,
        random.Random(1),
        on_game=lambda index, seat, game: given.append((index, seat, game)),
    )

    assert [(index, seat, game.payoffs, game.plies) for index, seat, game in given] == [
        (0, 0, (1.0, 0.0), 1),
        (1, 1, (0.0, 1.0), 1),
        (2, 0, (1.0, 0.0), 1),
    ]
