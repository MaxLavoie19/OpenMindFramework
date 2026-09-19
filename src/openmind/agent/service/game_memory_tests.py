import logging
from pathlib import Path

import pytest

from openmind.agent.constant.agent_constant import ARMS_GAME, DRAW, GAME_KEYWORD, LOSS, MODEL_KEYWORD, WIN
from openmind.agent.mapper.game_summary_json_mapper import GameSummaryJsonMapper
from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.game_memory import GameMemory
from openmind.knowledge.constant.knowledge_constant import DIRECT_EXPERIENCE, SELF_PLAY
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
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


def test_a_game_is_kept_as_a_direct_experience_with_each_model_once_and_each_player_s_outcome_drawn_from_it(tmp_path: Path) -> None:
    base = create_knowledge_base("chess", tmp_path)
    memory = GameMemory(base)

    memory.remember(game(1, (1.0, 0.0)))
    memory.remember(game(2, (0.5, 0.5), (WEIGHTED_ARM, WIN_ARM)))

    models = base.beliefs(tags=(("keyword", MODEL_KEYWORD),))
    assert [(belief.value, dict(belief.tags)["id"], dict(belief.tags)["name"]) for belief in models] == [
        (WIN_ARM.text, WIN_ARM.id, WIN_ARM.name),
        (WEIGHTED_ARM.text, WEIGHTED_ARM.id, WEIGHTED_ARM.name),
    ]
    games = base.experiences(tags=(("keyword", GAME_KEYWORD),))
    assert [(kept.source.mechanism, kept.source.parameter("game")) for kept in games] == [
        (base.mechanism_named(SELF_PLAY).id, "round 1 arms game 1"),  # type: ignore[union-attr]
        (base.mechanism_named(SELF_PLAY).id, "round 1 arms game 2"),  # type: ignore[union-attr]
    ]
    assert [belief.variable for belief in base.beliefs(tags=(("keyword", WIN),))] == ["outcome of white in round 1 arms game 1"]
    assert [belief.variable for belief in base.beliefs(tags=(("keyword", LOSS),))] == ["outcome of black in round 1 arms game 1"]
    assert [belief.variable for belief in base.beliefs(tags=(("keyword", DRAW),))] == [
        "outcome of white in round 1 arms game 2",
        "outcome of black in round 1 arms game 2",
    ]
    (drawn,) = base.beliefs(tags=(("keyword", WIN),))
    (evidence,) = drawn.evidence
    assert (drawn.certainty, evidence.source.mechanism, evidence.source.rests_on) == (
        1.0,
        base.mechanism_named(DIRECT_EXPERIENCE).id,  # type: ignore[union-attr]
        (games[0].id,),
    )
    chess = base.context_named("chess").id  # type: ignore[union-attr]
    assert base.belief("ending of round 1 arms game 1", chess).value == "threefold repetition"  # type: ignore[union-attr]


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
    assert len(base.beliefs(tags=(("keyword", MODEL_KEYWORD),))) == 2
    games = [
        GameSummaryJsonMapper().from_json(str(kept.value), reopened.models())
        for kept in base.experiences(tags=(("keyword", GAME_KEYWORD),))
    ]
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
