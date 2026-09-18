import logging
from pathlib import Path

import pytest

from openmind.agent.constant.agent_constant import ARMS_GAME, DRAW, GAME_KEYWORD, LOSS, MODEL_KEYWORD, WIN
from openmind.agent.mapper.game_summary_json_mapper import GameSummaryJsonMapper
from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.constant.doxastic_constant import PLAYED
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl

pytestmark = pytest.mark.log_level("INFO")

WIN_ARM = ModelDescription("win", '{"valuation": {"rules": ["win"]}}')
WEIGHTED_ARM = ModelDescription("weighted", '{"valuation": {"rules": ["weighted"]}}')


def game(number: int, payoffs: tuple[float, float], models: tuple[ModelDescription, ...] = (WIN_ARM, WEIGHTED_ARM)) -> GameSummary:
    return GameSummary(
        "chess",
        ARMS_GAME,
        1,
        number,
        (11, 12),
        ("white", "black"),
        models,
        payoffs,
        15,
        "threefold repetition",
        '[Result "1/2-1/2"] 1. Nh3 Nh6 1/2-1/2',
        TimeControl(60.0),
        (1.5, 2.0),
        (2.0, 2.1),
        (Clock(58.5), Clock(58.0)),
    )


def test_a_game_leaves_each_model_once_the_game_and_each_player_s_outcome(tmp_path: Path) -> None:
    base = create_knowledge_base("chess", tmp_path)
    memory = GameMemory(base)

    memory.remember(game(1, (1.0, 0.0)))
    memory.remember(game(2, (0.5, 0.5), (WEIGHTED_ARM, WIN_ARM)))

    models = base.recall(keyword=MODEL_KEYWORD)
    assert [(record.text, record.names) for record in models] == [
        (WIN_ARM.text, (WIN_ARM.id, WIN_ARM.name)),
        (WEIGHTED_ARM.text, (WEIGHTED_ARM.id, WEIGHTED_ARM.name)),
    ]
    assert len(base.recall(keyword=GAME_KEYWORD)) == 2
    assert [record.text for record in base.recall(keyword=WIN)] == ["win won as white in round 1 arms game 1"]
    assert [record.text for record in base.recall(keyword=LOSS)] == ["weighted lost as black in round 1 arms game 1"]
    assert [record.text for record in base.recall(keyword=DRAW)] == [
        "weighted drew as white in round 1 arms game 2",
        "win drew as black in round 1 arms game 2",
    ]
    assert {(record.provenance.source, record.provenance.game, record.provenance.round) for record in base.recall()} == {
        (PLAYED, "round 1 arms game 1", 1),
        (PLAYED, "round 1 arms game 2", 1),
    }


def test_a_model_is_logged_the_first_time_it_plays(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.agent.service.game_memory")
    memory = GameMemory(create_knowledge_base("chess", tmp_path))

    memory.remember(game(1, (1.0, 0.0)))
    memory.remember(game(2, (1.0, 0.0)))

    assert caplog.messages == [
        f"Remembered model {WIN_ARM.id} (win)",
        f"Remembered model {WEIGHTED_ARM.id} (weighted)",
    ]


def test_scores_count_wins_draws_and_losses_by_model_name_or_id(tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("chess", tmp_path))
    for number, payoffs in enumerate(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (1.0, 0.0)), start=1):
        memory.remember(game(number, payoffs))

    assert memory.scores((WIN_ARM.name, WEIGHTED_ARM.id, "never played")) == {
        WIN_ARM.name: (4, 2, 1, 1),
        WEIGHTED_ARM.id: (4, 1, 1, 2),
        "never played": (0, 0, 0, 0),
    }


def test_after_a_crash_the_knowledge_base_finds_every_finished_game_and_its_models(tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("chess", tmp_path))
    memory.remember(game(1, (1.0, 0.0)))
    memory.remember(game(2, (0.5, 0.5)))

    reopened = GameMemory(create_knowledge_base("chess", tmp_path))
    reopened.remember(game(3, (0.0, 1.0)))

    base = create_knowledge_base("chess", tmp_path)
    assert reopened.models() == (WIN_ARM, WEIGHTED_ARM)
    assert len(base.recall(keyword=MODEL_KEYWORD)) == 2
    games = [GameSummaryJsonMapper().from_json(record.text, reopened.models()) for record in base.recall(keyword=GAME_KEYWORD)]
    assert games == [game(1, (1.0, 0.0)), game(2, (0.5, 0.5)), game(3, (0.0, 1.0))]


def test_remembered_games_come_back_in_the_order_they_ended_and_are_counted_by_kind(tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("chess", tmp_path))
    memory.remember(game(1, (1.0, 0.0)))
    memory.remember(game(2, (0.5, 0.5)))

    assert GameMemory(create_knowledge_base("chess", tmp_path)).games("arms") == (game(1, (1.0, 0.0)), game(2, (0.5, 0.5)))
    assert (memory.count("arms"), memory.count("match")) == (2, 0)


def test_new_games_are_numbered_after_the_highest_number_remembered_whatever_how_many(tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("chess", tmp_path))
    assert memory.last_number("arms") == 0

    memory.remember(game(6, (1.0, 0.0)))

    assert (memory.count("arms"), memory.last_number("arms")) == (1, 6)
