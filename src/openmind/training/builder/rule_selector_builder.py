from typing import Self

from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.evaluation.service.choice_measurer import ChoiceMeasurer
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.evaluation.service.reference_search import ReferenceSearch
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.training.service.non_inferiority_test import NonInferiorityTest
from openmind.training.service.rule_selector import RuleSelector
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.service.state_reader import StateReader


class RuleSelectorBuilder:
    """Sets how many worker processes a rule selector searches in, 1 by default, and wires the services it selects
    with."""

    def __init__(self) -> None:
        self._workers = 1

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def build(self) -> RuleSelector:
        solver = SolverBuilder().build()
        predictor = PredictorBuilder().build()
        return RuleSelector(
            ExactSearch(solver, predictor, StateReader()),
            ReferenceSearch(solver, predictor),
            ChoiceMeasurer(StateTextMapper(), ActionTextMapper()),
            TaskRunner(self._workers),
            NonInferiorityTest(),
            RuleTextMapper(),
        )
