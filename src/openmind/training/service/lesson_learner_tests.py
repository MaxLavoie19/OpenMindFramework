import json
from pathlib import Path

import numpy as np

from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.inference.model.deduction import Deduction
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rule.model.python_rule import PythonRule
from openmind.training.constant.continuous_constant import PROOF_KEYWORD
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.game_lesson import GameLesson
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.pondering import Pondering
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_readings import SignalReadings
from openmind.training.model.signal_settings import SignalSettings
from openmind.training.service.lesson_learner import LessonLearner
from openmind.training.service.online_value_fitter import OnlineValueFitter
from openmind.training.service.signal_recorder import SignalRecorder
from openmind.world.model.state import State

PIECES = PythonRule("pieces")
SETTINGS = ContinuousTrainingSettings(
    None, 10, 1, ValueSettings((0.1, 0.01), 10, 1e-6, 1.0, 1024**3), SignalSettings(), 0, None, None, learning_rate=0.5
)


def lesson(targets: dict[str, np.ndarray]) -> GameLesson:
    first, second = State((("at", 1),)), State((("at", 2),))
    game = PlayedGame((), (first, second), (0.5, 0.5), (1.0, 0.0), ("pieces", "deduced"), ())
    proof = Deduction(second, "white", None, (1.0, 0.0), (), 2)
    pondering = Pondering((), (proof,), (), ())
    readings = SignalReadings(2, 1, 1, 1, {"pieces": np.array([1.0, 1.0])})
    return GameLesson(game, (Signal("pieces", PIECES),), readings, pondering, targets, {PIECES: np.array([1.0, -1.0, 1.0, -1.0])})


def library() -> SignalLibrary:
    base = ValueBase("chess", 0.0, 0.0, 1.0, (ValueRule(PIECES, 0.1),))
    return SignalLibrary("chess", value_bases=(("pieces", base), ("deduced", base)))


def test_a_lesson_adds_to_the_records_steps_every_arm_and_keeps_the_library_s_arms(tmp_path: Path) -> None:
    learner = LessonLearner(SignalRecorder(None), OnlineValueFitter())  # type: ignore[arg-type]
    rising, falling = np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0])

    learned = learner.learn(DOMAIN, library(), lesson({"pieces": rising, "weighted": falling}), "arms game 1", SETTINGS)

    (record,) = learned.records
    assert (record.signal.name, record.agreements) == ("pieces", 2)
    weights = {name: base.rules[0].weight for name, base in learned.value_bases}
    # pieces follows its own targets and grows; deduced follows the weighted aggregation's and shrinks
    assert weights["pieces"] > 0.1 > weights["deduced"]


def test_a_lesson_s_proofs_join_the_pool_and_are_remembered_with_their_game_and_ply(tmp_path: Path) -> None:
    base = create_knowledge_base("chess", tmp_path)
    learner = LessonLearner(SignalRecorder(None), OnlineValueFitter(), base)  # type: ignore[arg-type]

    learner.learn(DOMAIN, library(), lesson({"win": np.array([1.0, 0.0, 1.0, 0.0])}), "arms game 7", SETTINGS)

    assert len(learner.deductions) == 1
    (proof,) = base.recall(keyword=PROOF_KEYWORD)
    assert json.loads(proof.text) == {"game": "arms game 7", "ply": 1, "player": "white", "payoffs": [1.0, 0.0]}
    assert (proof.provenance.source, proof.provenance.game, proof.provenance.ply) == ("proved", "arms game 7", 1)


class _Domain:
    name = "chess"


DOMAIN = _Domain()
