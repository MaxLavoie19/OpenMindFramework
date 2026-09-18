from collections.abc import Callable
import logging
from dataclasses import replace

import pytest

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.factory.training_factory import create_value_training_loop
from openmind.training.model.training_report import TrainingReport
from openmind.training.model.value_distillation_settings import ValueDistillationSettings
from openmind.training.model.value_training_settings import ValueTrainingSettings

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]

SETTINGS = ValueTrainingSettings(
    rounds=2,
    distillation=ValueDistillationSettings(
        games=2,
        held_out_games=1,
        iterations=10,
        seed=1,
        target="search",
        values=ValueSettings(
            prices=(0.1, 0.01), max_steps=100, tolerance=1e-6, seconds=300.0, memory_bytes=1024**3, candidates=1000
        ),
    ),
    rollout_actions=1,
    rollout_limit=None,
    unfinished_payoff=None,
    evaluation_games=2,
    start_file=None,
)


def test_a_round_carries_what_the_game_records_of_its_games(game: Game, knowledge: KnowledgeBase) -> None:
    rbs = game("tictactoe")
    RuleDeclarer(knowledge, rbs.context).record(
        PythonRule("' '.join(f\"{dict(action.parameters)['row']}{dict(action.parameters)['col']}\" for action in actions)")
    )
    recording = create_rule_based_system(knowledge, rbs.context)
    settings = replace(SETTINGS, rounds=1, evaluation_games=0)

    (round_one,) = create_value_training_loop(knowledge).train(recording, None, settings).rounds

    assert len(round_one.records) == settings.distillation.games
    assert all(record and "\n" not in record for record in round_one.records)
    assert not create_value_training_loop(knowledge).train(game("sudoku"), None, settings).rounds[0].records


def test_each_round_self_plays_with_the_previous_round_s_rules_and_plays_its_opponents(knowledge: KnowledgeBase, game: Game, 
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    reports: list[TrainingReport] = []

    report = create_value_training_loop(knowledge).train(game("tictactoe"), None, SETTINGS, reports.append)

    assert [item.number for item in report.rounds] == [1, 2]
    assert [[(results.opponent, results.games) for results in item.baselines] for item in report.rounds] == [
        [("random", 2), ("untrained MCTS", 2)],
        [("random", 2), ("untrained MCTS", 2)],
    ]
    assert report.rounds[0].against_previous is None
    assert report.rounds[1].against_previous is not None
    assert (report.rounds[1].against_previous.opponent, report.rounds[1].against_previous.games) == ("round 1", 2)
    assert all(item.training_rows > 0 and item.fits for item in report.rounds)
    assert [(len(item.rounds), item.complete) for item in reports] == [(1, False), (1, False), (2, False), (2, True)]
    assert (reports[0].rounds[0].baselines, reports[0].rounds[0].context) == ((), report.rounds[0].context)
    assert reports[-1] == report
    assert "Round 1 fitted: handing it over before its games" in caplog.messages
    assert "Round 1 of 2: self-play without value rules" in caplog.messages
    assert "Round 2 of 2: self-play valuing positions with round 1's rules" in caplog.messages
    assert any(message.startswith("Round 2 against round 1: 2 games, ") for message in caplog.messages)


def test_round_1_starts_from_the_start_rules_and_plays_their_agent(
    game: Game, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    rbs = game("tictactoe")
    start = RuleDeclarer(knowledge, "start rules")
    start.inherits(rbs.context)
    start.position("wins(me)", PythonRule("wins(me)"), 1.0)

    report = create_value_training_loop(knowledge).train(
        rbs, "start rules", replace(SETTINGS, rounds=1, start_file="start.json")
    )

    assert report.rounds[0].against_previous is not None
    assert report.rounds[0].against_previous.opponent == "start rules"
    assert report.complete
    assert "Round 1 of 1: self-play valuing positions with the start rules" in caplog.messages


def test_with_a_deduction_budget_self_play_deduces_where_it_has_no_rules(knowledge: KnowledgeBase, game: Game, 
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind")
    budget = DeductionBudget(1, 30.0)
    settings = replace(
        SETTINGS,
        rounds=1,
        evaluation_games=0,
        deduction=budget,
    )

    report = create_value_training_loop(knowledge).train(game("tictactoe"), None, settings)

    assert report.rounds[0].fits
    assert any(message.startswith("Deduced ") for message in caplog.messages)


def test_without_evaluation_games_a_round_plays_no_series(knowledge: KnowledgeBase, game: Game) -> None:
    report = create_value_training_loop(knowledge).train(
        game("tictactoe"), None, replace(SETTINGS, rounds=1, evaluation_games=0)
    )

    assert (report.rounds[0].baselines, report.rounds[0].against_previous) == ((), None)
