from typing import Self

from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.builder.rule_generator_builder import RuleGeneratorBuilder
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.service.distiller import Distiller
from openmind.training.service.self_play import SelfPlay
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class DistillerBuilder:
    """Sets how many worker processes a distiller's self-play games and rule condition checks run in, 1 by default, and
    wires the services it works with."""

    def __init__(self) -> None:
        self._workers = 1

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def build(self) -> Distiller:
        self_play = SelfPlay(SolverBuilder().build(), PredictorBuilder().build(), StateReader(), TaskRunner(self._workers))
        return Distiller(
            self_play,
            RuleGeneratorBuilder().with_workers(self._workers).build(),
            RuleCompiler(),
            RuleRunner(StateNamespaceMapper(VariableNameMapper())),
            ConsequenceLibraryBuilder().build(),
        )
