import logging
import random
import re
from dataclasses import replace

import pytest

from openmind.rule.model.python_rule import PythonRule

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.service.timekeeper import Timekeeper
from openmind.agent.service.timekeeper_tests import TIMEOUT, alternating_steps
from openmind.csp.factory.csp_factory import create_solver
from openmind.parallel.service.task_runner import TaskRunner
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.training.model.played_game import PlayedGame
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.self_play import SelfPlay
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

GAME_LINE = re.compile(
    r"Self-play game with seeds \d+ and \d+ finished in (?P<plies>\d+) plies: (?P<samples>\d+) samples, payoffs (?P<payoffs>.*)"
)


def new_self_play(workers: int = 1) -> SelfPlay:
    return SelfPlay(create_solver(), create_predictor(), StateReader(), TaskRunner(workers))


def test_records_give_what_the_domain_records_of_each_game() -> None:
    plain = create_tictactoe_domain()
    recording = replace(
        plain,
        record=PythonRule("' '.join(f\"{dict(action.parameters)['row']}{dict(action.parameters)['col']}\" for action in actions)"),
    )
    self_play = new_self_play()
    games = self_play.play(plain, AgentBuilder().with_iterations(10).with_exploration(1.4), 2, random.Random(1))

    records = self_play.records(recording, games)

    assert len(records) == 2
    assert [len(record.split()) for record in records] == [len(game.actions) for game in games]
    assert self_play.records(plain, games) == ()


def test_play_keeps_every_game_with_its_searches_positions_and_payoffs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    domain = create_tictactoe_domain()

    games = new_self_play().play(domain, AgentBuilder().with_iterations(10).with_exploration(1.4), 2, random.Random(1))

    game_lines = [
        GAME_LINE.fullmatch(record.getMessage())
        for record in caplog.records
        if record.name == "openmind.training.service.self_play"
    ]
    assert all(game_lines) and len(game_lines) == 2
    assert [(int(line["plies"]), int(line["samples"])) for line in game_lines] == [  # type: ignore[index]
        (len(game.states), len(game.samples)) for game in games
    ]
    assert [line["payoffs"] for line in game_lines] == [  # type: ignore[index]
        " ".join(f"{name}={payoff}" for name, payoff in zip(domain.players.names, game.payoffs)) for game in games
    ]
    assert {sample.action.name for game in games for sample in game.samples} == {"place"}
    for game in games:
        assert game.states[0] == domain.initial_state
        assert 5 <= len(game.states) == len(game.search_values) <= 9
        assert all(0.0 <= value <= 1.0 for value in game.search_values)
        assert sum(game.payoffs) == 1.0


def test_games_played_without_keeping_samples_keep_their_positions_and_count_their_samples(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    domain = create_tictactoe_domain()

    (game,) = new_self_play().play(
        domain, AgentBuilder().with_iterations(10).with_exploration(1.4), 1, random.Random(1), keep_samples=False
    )

    (line,) = [
        GAME_LINE.fullmatch(record.getMessage())
        for record in caplog.records
        if record.name == "openmind.training.service.self_play"
    ]
    assert game.samples == () and len(game.states) == len(game.search_values) >= 5
    assert line is not None and int(line["samples"]) > 0


def test_arms_play_each_other_each_player_s_agent_following_its_arm_and_scores_add_up_as_games_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    domain = create_tictactoe_domain()
    builders = {arm: AgentBuilder().with_iterations(iterations).with_exploration(1.4) for arm, iterations in (("few", 5), ("many", 30), ("some", 10))}

    games = new_self_play().play_arms(domain, builders, 4, {"many": (2, 2.0)}, ArmSelector(), 1.4, random.Random(1))

    assert len(games) == 4
    assert all(len(set(game.arms)) == 2 and set(game.arms) <= set(builders) for game in games)
    assert "many" not in games[0].arms
    arms_lines = [message for message in caplog.messages if message.startswith("Arms game ")]
    assert len(arms_lines) == 4 and " scores " in arms_lines[0]
    assert all(" against " in message for message in caplog.messages if message.startswith("Self-play game with seeds "))


def test_arms_need_a_two_player_domain() -> None:
    from dataclasses import replace

    from openmind.world.model.players import Players

    alone = replace(create_tictactoe_domain(), players=Players(("X",), "turn", ("payoff(X)",)))
    builders = {arm: AgentBuilder().with_iterations(5).with_exploration(1.4) for arm in ("a", "b")}

    with pytest.raises(ValueError, match="two players, not 1"):
        new_self_play().play_arms(alone, builders, 1, {}, ArmSelector(), 1.4, random.Random(1))


def test_a_domain_saying_why_games_end_and_recording_them_gets_both_logged(caplog: pytest.LogCaptureFixture) -> None:
    from dataclasses import replace

    from openmind.rule.model.python_rule import PythonRule

    caplog.set_level(logging.INFO)
    domain = replace(create_tictactoe_domain(), ending=PythonRule("'the end'"), record=PythonRule("len(actions)"))

    (game,) = new_self_play().play(domain, AgentBuilder().with_iterations(10).with_exploration(1.4), 1, random.Random(1))

    assert len(game.actions) == len(game.states) >= 5
    assert any(" plies by the end: " in message for message in caplog.messages)
    assert any(message.endswith(f" record: {len(game.actions)}") for message in caplog.messages)


def test_workers_play_the_same_games() -> None:
    domain = create_tictactoe_domain()

    alone, together = (
        new_self_play(workers).play(domain, AgentBuilder().with_iterations(10).with_exploration(1.4), 3, random.Random(1))
        for workers in (1, 2)
    )

    assert alone == together


def clocked_self_play() -> SelfPlay:
    """Self-play whose referee reads a time source moving on by a second a reading: every move takes a second."""
    return SelfPlay(
        create_solver(), create_predictor(), StateReader(), TaskRunner(1), timekeeper=Timekeeper(create_rule_caller(), Ticking())
    )


def timed_builder() -> AgentBuilder:
    return AgentBuilder().with_iterations(5).with_exploration(1.4).with_time_budget_estimator(PlainTimeBudgetEstimator(30))


def test_on_a_clock_each_move_is_charged_to_its_player_and_the_increment_added(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    domain = alternating_steps(TIMEOUT)

    (game,) = clocked_self_play().play(domain, timed_builder(), 1, random.Random(1), time_control=TimeControl(1.5, 1.0))

    assert game.payoffs == (0.5, 0.5) and game.flagged is None
    assert game.time_control == TimeControl(1.5, 1.0)
    assert game.seconds == (1.0, 1.0, 1.0, 1.0)
    assert len(game.budgets) == 4 and all(budget is not None and budget > 0.0 for budget in game.budgets)
    assert game.clocks == (Clock(1.5, 1.0), Clock(1.5, 1.0))
    assert any(re.search(r"finished in 4 plies on 0\.025\+1, clocks A=1\.50 B=1\.50:", message) for message in caplog.messages)


def test_a_player_whose_time_runs_out_loses_by_the_timeout_rule_without_playing_the_move(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    domain = alternating_steps(TIMEOUT)

    (game,) = clocked_self_play().play(domain, timed_builder(), 1, random.Random(1), time_control=TimeControl(1.5))

    assert game.flagged == "A" and game.payoffs == (0.0, 1.0)
    assert len(game.actions) == len(game.states) == 2 and len(game.seconds) == 3
    assert game.clocks == (Clock(-0.5, 0.0, True), Clock(0.5))
    assert any("finished in 2 plies by A's flag on 0.025+0, clocks A=-0.50 B=0.50:" in message for message in caplog.messages)


def test_arms_play_on_a_clock_too() -> None:
    domain = alternating_steps(TIMEOUT)
    builders = {"first": timed_builder(), "second": timed_builder()}

    (game,) = clocked_self_play().play_arms(
        domain, builders, 1, {}, ArmSelector(), 1.4, random.Random(1), time_control=TimeControl(1.5)
    )

    assert game.flagged == "A" and set(game.arms) == {"first", "second"}


def test_a_domain_without_a_timeout_rule_can_t_be_played_on_a_clock() -> None:
    with pytest.raises(ValueError, match="no timeout rule"):
        clocked_self_play().play(alternating_steps(None), timed_builder(), 1, random.Random(1), time_control=TimeControl(60.0))


def test_each_game_is_given_as_it_ends_with_its_seeds_and_ending() -> None:
    given: list[tuple[int, PlayedGame]] = []
    domain = alternating_steps(TIMEOUT)

    games = new_self_play().play(
        domain, AgentBuilder().with_iterations(5).with_exploration(1.4), 3, random.Random(1), on_game=lambda index, game: given.append((index, game))
    )

    assert [index for index, _ in given] == [0, 1, 2]
    assert tuple(game for _, game in given) == games
    assert all(game.agent_seed is not None and game.outcome_seed is not None for game in games)


def test_an_arms_game_is_given_before_its_result_adds_to_the_scores(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    domain = create_tictactoe_domain()
    builders = {"first": AgentBuilder().with_iterations(5).with_exploration(1.4), "second": AgentBuilder().with_iterations(5).with_exploration(1.4)}
    logger = logging.getLogger("test.on_game")

    new_self_play().play_arms(
        domain, builders, 2, {}, ArmSelector(), 1.4, random.Random(1), on_game=lambda index, game: logger.info("given %d", index)
    )

    messages = [message for message in caplog.messages if message.startswith(("given", "Arms game"))]
    assert [message.split(":")[0] for message in messages] == ["given 0", "Arms game 1", "given 1", "Arms game 2"]
