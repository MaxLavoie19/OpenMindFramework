import logging
from collections.abc import Sequence

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rule.model.python_rule import PythonRule
from openmind.training.constant.signal_constant import UNIFORM, WEIGHTED, WIN
from openmind.training.constant.training_constant import SEARCH_TARGET
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.game_lesson import GameLesson
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.model.signal import Signal
from openmind.training.model.study_snapshot import StudySnapshot
from openmind.training.service.candidate_signals import CandidateSignals
from openmind.training.service.position_ponderer import PositionPonderer
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.signal_ranker import SignalRanker
from openmind.training.service.signal_recorder import SignalRecorder
from openmind.training.service.signal_targeter import SignalTargeter

logger = logging.getLogger(__name__)


class GameStudy:
    """What a worker does with one call of continuous training: plays a game between two arms, then studies it before
    taking the next one.

    1. It ponders the game, when the settings say so: the positions the reference rules missed most, and, for a decisive
       game, its positions walked back from the end.
    2. It reads every candidate signal on the game's anchors, the proven positions taking their proven winner.
    3. It works out the targets of the signals followed, winning, the best recorded signals and their uniform and
       weighted aggregations, on every position of the game for each player, proven positions at their proven payoffs.
    4. It reads every term of the arms' value rules on those same rows, for the weight step the main process takes.

    Every service it holds works in this process: its term evaluator and ponderer need a task runner of one worker."""

    def __init__(
        self,
        self_play: SelfPlay,
        position_ponderer: PositionPonderer,
        signal_recorder: SignalRecorder,
        signal_targeter: SignalTargeter,
        signal_ranker: SignalRanker,
        candidate_signals: CandidateSignals,
        term_evaluator: TermEvaluator,
        position_row_mapper: PositionRowMapper,
    ) -> None:
        self._position_row_mapper = position_row_mapper
        self._self_play = self_play
        self._position_ponderer = position_ponderer
        self._signal_recorder = signal_recorder
        self._signal_targeter = signal_targeter
        self._signal_ranker = signal_ranker
        self._candidate_signals = candidate_signals
        self._term_evaluator = term_evaluator

    def play_and_study(
        self,
        domain: Domain,
        builders: Sequence[AgentBuilder],
        arms: tuple[str, ...],
        agent_seed: int,
        outcome_seed: int,
        snapshot: StudySnapshot,
        settings: ContinuousTrainingSettings,
    ) -> GameLesson:
        game = self._self_play.play_arm_game(
            domain, builders, arms, agent_seed, outcome_seed, keep_samples=False, time_control=settings.time_control
        )
        return self.study(domain, game, snapshot, settings)

    def study(
        self, domain: Domain, game: PlayedGame, snapshot: StudySnapshot, settings: ContinuousTrainingSettings
    ) -> GameLesson:
        library = snapshot.library
        pondering = None
        if settings.deduction is not None and (settings.ponder_positions or settings.ponder_endings):
            rows = self._position_row_mapper.to_rows(domain, (game,), SEARCH_TARGET)
            pondering = self._position_ponderer.ponder(
                domain,
                rows,
                snapshot.reference,
                PonderingSettings(settings.ponder_positions, settings.deduction, settings.ponder_endings),
                (game,),
            )
        deductions = (
            ()
            if pondering is None
            else (*pondering.deductions, *(deduction for walk in pondering.walks for deduction in walk.deductions))
        )
        seeds = (*snapshot.seeds, *(() if pondering is None else pondering.seeds))
        followed = (Signal(WIN), *(record.signal for record in self._signal_ranker.best(library, settings.signals.arms)))
        parts = tuple(signal.name for signal in followed)
        aggregations = (Signal(UNIFORM, parts=parts), Signal(WEIGHTED, parts=parts))
        candidates = (*self._candidate_signals.candidates(domain, (game,), seeds, library), Signal(WIN), *aggregations)
        readings = self._signal_recorder.read(domain, candidates, (game,), deductions)
        reliabilities = self._signal_ranker.reliabilities(library, followed)
        readings = self._signal_recorder.aggregate(readings, aggregations[1], reliabilities)
        targets = self._signal_targeter.targets(
            domain, (game,), (*followed, *aggregations), reliabilities, settings.signals.horizon, deductions
        )
        terms = tuple(dict.fromkeys(rule.term for _, base in library.value_bases for rule in base.rules))
        rows = self._signal_targeter.rows(domain, (game,))
        columns = self._term_evaluator.columns(domain, rows, terms) if terms and rows else []
        return GameLesson(game, candidates, readings, pondering, targets, dict(zip(terms, columns, strict=True)))
