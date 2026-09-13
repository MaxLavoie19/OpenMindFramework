import logging
import random

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.training.service.self_play import SelfPlay
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


def test_play_collects_the_samples_of_every_search(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    names = VariableNameMapper()
    interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
    self_play = SelfPlay(
        create_solver(),
        Predictor(interpreter, names, expression_text, action_text),
        StateReader(),
    )

    samples = self_play.play(
        create_tictactoe_domain(), AgentBuilder().with_iterations(10).with_exploration(1.4), 2, random.Random(1)
    )

    game_lines = [
        record.getMessage() for record in caplog.records if record.name == "openmind.training.service.self_play"
    ]
    assert [line.split(":")[0] for line in game_lines] == ["Self-play game 1", "Self-play game 2"]
    assert sum(int(line.split(": ")[1].split(" ")[0]) for line in game_lines) == len(samples)
    assert {sample.action.name for sample in samples} == {"place"}
