from openmind.agent.model.model_description import ModelDescription
from openmind.evaluation.mapper.match_game_summary_mapper import MatchGameSummaryMapper
from openmind.evaluation.model.match_game import MatchGame
from openmind.evaluation.service.match_runner_tests import Declare, first_mover_decides
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl

EVALUATED = ModelDescription("round 2", '{"iterations": 100}')
OPPONENT = ModelDescription("random", '{"policy": "uniformly random legal actions"}')


def test_the_evaluated_model_sits_in_its_seat_and_a_match_s_steps_have_no_budget(declared: Declare) -> None:
    game = MatchGame((0.0, 1.0), None, 1, None, (), 7, 8, (0.5,), (Clock(59.5), Clock(60.0)))

    summary = MatchGameSummaryMapper().to_summary(
        first_mover_decides(declared), game, "match", 2, 4, 1, EVALUATED, OPPONENT, None, TimeControl(60.0)
    )

    assert summary.models == (OPPONENT, EVALUATED)
    assert summary.players == ("A", "B") and summary.seeds == (7, 8)
    assert summary.budgets == (None,) and summary.label == "round 2 match game 4"
