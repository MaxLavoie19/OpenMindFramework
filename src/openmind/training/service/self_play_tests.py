import logging
import random

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.training.service.self_play import SelfPlay
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


def new_self_play(workers: int = 1) -> SelfPlay:
    return SelfPlay(create_solver(), create_predictor(), StateReader(), TaskRunner(workers))


def test_play_collects_the_samples_of_every_search(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    samples = new_self_play().play(
        create_tictactoe_domain(), AgentBuilder().with_iterations(10).with_exploration(1.4), 2, random.Random(1)
    )

    game_lines = [
        record.getMessage() for record in caplog.records if record.name == "openmind.training.service.self_play"
    ]
    assert [line.split(":")[0] for line in game_lines] == ["Self-play game 1", "Self-play game 2"]
    assert sum(int(line.split(": ")[1].split(" ")[0]) for line in game_lines) == len(samples)
    assert {sample.action.name for sample in samples} == {"place"}


def test_workers_play_the_same_games() -> None:
    domain = create_tictactoe_domain()

    alone, together = (
        new_self_play(workers).play(domain, AgentBuilder().with_iterations(10).with_exploration(1.4), 3, random.Random(1))
        for workers in (1, 2)
    )

    assert alone == together
