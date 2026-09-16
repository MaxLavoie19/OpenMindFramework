import logging
from dataclasses import replace

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rule.model.python_rule import PythonRule
from openmind.training.factory.training_factory import create_value_training_loop
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_settings import SignalSettings
from openmind.training.model.training_report import TrainingReport
from openmind.training.model.value_distillation_settings import ValueDistillationSettings
from openmind.training.model.value_training_settings import ValueTrainingSettings

pytestmark = pytest.mark.log_level("INFO")

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


def test_with_the_signals_target_the_signal_library_carries_from_round_to_round() -> None:
    distillation = replace(SETTINGS.distillation, games=3, target="signals", signals=SignalSettings(arms=2))
    settings = replace(SETTINGS, evaluation_games=0, distillation=distillation)

    report = create_value_training_loop().train(create_tictactoe_domain(), None, settings)

    first, second = report.rounds
    assert first.library is not None and second.library is not None

    def readings(library: SignalLibrary) -> int:
        return sum(record.agreements + record.disagreements for record in library.records)

    assert readings(second.library) >= readings(first.library)
    assert [record.signal.name for record in second.arms][-2:] == ["uniform", "weighted"]
    assert len(first.library.value_bases) >= 3
    # Round 1's arms are the deduced value bases, round 2's the signals round 1 fitted: every game scores two arms.
    assert sum(record.games for record in first.library.records) == 2 * (3 + 1)
    assert sum(record.games for record in second.library.records) == 2 * 2 * (3 + 1)


def test_with_the_signals_target_a_library_without_value_bases_is_prepared_and_one_with_them_is_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    distillation = replace(SETTINGS.distillation, games=3, target="signals", signals=SignalSettings(arms=2))
    settings = replace(SETTINGS, rounds=1, evaluation_games=0, distillation=distillation)
    loop, domain = create_value_training_loop(), create_tictactoe_domain()

    (prepared,) = loop.train(domain, None, settings).rounds
    kept_bases = prepared.library.value_bases  # type: ignore[union-attr]
    kept = SignalLibrary("tictactoe", (), (), (("mine", kept_bases[0][1]), ("yours", kept_bases[1][1])))
    caplog.clear()
    (loaded,) = loop.train(domain, None, settings, library=kept).rounds

    first_records = {record.signal.name: record for record in prepared.library.records}  # type: ignore[union-attr]
    assert {"options", "my options", "their options"} <= first_records.keys() and first_records["options"].signal.premises == ()
    assert sum(first_records[name].games for name in first_records if name.startswith("deduced")) == 2 * (3 + 1)
    loaded_records = {record.signal.name: record for record in loaded.library.records}  # type: ignore[union-attr]
    assert (loaded_records["mine"].games + loaded_records["yours"].games, "options" in loaded_records) == (2 * (3 + 1), False)
    assert not any(message.startswith("Prepared ") for message in caplog.messages)


def test_a_round_carries_what_the_domain_records_of_its_games() -> None:
    recording = replace(
        create_tictactoe_domain(),
        record=PythonRule("' '.join(f\"{dict(action.parameters)['row']}{dict(action.parameters)['col']}\" for action in actions)"),
    )
    settings = replace(SETTINGS, rounds=1, evaluation_games=0)

    (round_one,) = create_value_training_loop().train(recording, None, settings).rounds

    assert len(round_one.records) == settings.distillation.games
    assert all(record and "\n" not in record for record in round_one.records)
    assert not create_value_training_loop().train(create_tictactoe_domain(), None, settings).rounds[0].records


def test_each_round_self_plays_with_the_previous_round_s_rules_and_plays_its_opponents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    reports: list[TrainingReport] = []

    report = create_value_training_loop().train(create_tictactoe_domain(), None, SETTINGS, reports.append)

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
    assert (reports[0].rounds[0].baselines, reports[0].rounds[0].value_base) == ((), report.rounds[0].value_base)
    assert reports[-1] == report
    assert "Round 1 fitted: handing it over before its games" in caplog.messages
    assert "Round 1 of 2: self-play without value rules" in caplog.messages
    assert "Round 2 of 2: self-play valuing positions with round 1's rules" in caplog.messages
    assert any(message.startswith("Round 2 against round 1: 2 games, ") for message in caplog.messages)


def test_round_1_starts_from_the_start_rules_and_plays_their_agent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    start = ValueBase("tictactoe", 0.0, 0.0, 1.0, (ValueRule(PythonRule("wins(me)"), 1.0),))

    report = create_value_training_loop().train(
        create_tictactoe_domain(), start, replace(SETTINGS, rounds=1, start_file="start.json")
    )

    assert report.rounds[0].against_previous is not None
    assert report.rounds[0].against_previous.opponent == "start rules"
    assert report.complete
    assert "Round 1 of 1: self-play valuing positions with the start rules" in caplog.messages


def test_with_a_deduction_budget_self_play_deduces_where_it_has_no_rules_and_the_round_ponders(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind")
    budget = DeductionBudget(1, 30.0)
    settings = replace(
        SETTINGS,
        rounds=1,
        evaluation_games=0,
        deduction=budget,
        distillation=replace(SETTINGS.distillation, pondering=PonderingSettings(2, budget)),
    )

    report = create_value_training_loop().train(create_tictactoe_domain(), None, settings)

    pondering = report.rounds[0].pondering
    assert pondering is not None and pondering.positions == 2
    assert any(message.startswith("Deduced ") for message in caplog.messages)


def test_without_evaluation_games_a_round_plays_no_series() -> None:
    report = create_value_training_loop().train(
        create_tictactoe_domain(), None, replace(SETTINGS, rounds=1, evaluation_games=0)
    )

    assert (report.rounds[0].baselines, report.rounds[0].against_previous) == ((), None)
