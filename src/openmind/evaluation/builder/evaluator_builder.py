from typing import Self

from openmind.agent.service.game_memory import GameMemory

from openmind.evaluation.service.choice_measurer import ChoiceMeasurer
from openmind.evaluation.service.evaluator import Evaluator
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.evaluation.service.reference_search import ReferenceSearch
from openmind.evaluation.service.value_measurer import ValueMeasurer
from openmind.parallel.service.task_runner import TaskRunner
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.service.state_reader import StateReader


class EvaluatorBuilder:
    """Sets how many worker processes an evaluator runs games and searches in, 1 by default, and wires the services it
    measures with."""

    def __init__(self) -> None:
        self._workers = 1
        self._game_memory: GameMemory | None = None

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def with_game_memory(self, game_memory: GameMemory | None) -> Self:
        """Where every match game is remembered as it ends; None, the default, remembers none."""
        self._game_memory = game_memory
        return self

    def build(self) -> Evaluator:
        state_reader = StateReader()
        task_runner = TaskRunner(self._workers)
        state_text_mapper, action_text_mapper = StateTextMapper(), ActionTextMapper()
        choice_measurer = ChoiceMeasurer(state_text_mapper, action_text_mapper)
        return Evaluator(
            MatchRunner(state_reader, task_runner),
            ExactSearch(state_reader),
            ReferenceSearch(),
            choice_measurer,
            ValueMeasurer(state_reader, choice_measurer),
            task_runner,
            state_text_mapper,
            action_text_mapper,
            self._game_memory,
        )
