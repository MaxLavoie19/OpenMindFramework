from typing import Self

from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.builder.value_distiller_builder import ValueDistillerBuilder
from openmind.training.service.heuristic_deducer import HeuristicDeducer
from openmind.training.service.signal_preparer import SignalPreparer
from openmind.training.service.value_training_loop import ValueTrainingLoop
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class ValueTrainingLoopBuilder:
    """Sets how many worker processes a value training loop's self-play, term evaluations and games run in, 1 by
    default, and the memory each of them holds at most, no cap by default, and wires the loop: its value distiller,
    match runner, and the rule services its agents value with."""

    def __init__(self) -> None:
        self._workers = 1
        self._memory_cap: MemoryCap | None = None

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def with_memory_cap(self, memory_cap: MemoryCap | None) -> Self:
        self._memory_cap = memory_cap
        return self

    def build(self) -> ValueTrainingLoop:
        names = VariableNameMapper()
        mechanics = Mechanics(SolverBuilder().build(), PredictorBuilder().build(), StateNamespaceMapper(names), MemoryMeter())
        return ValueTrainingLoop(
            ValueDistillerBuilder().with_workers(self._workers).with_memory_cap(self._memory_cap).build(),
            MatchRunner(
                SolverBuilder().build(),
                PredictorBuilder().build(),
                StateReader(),
                TaskRunner(self._workers, self._memory_cap),
            ),
            RuleCompiler(),
            RuleRunner(StateNamespaceMapper(VariableNameMapper())),
            ConsequenceLibraryBuilder().build(),
            SignalPreparer(HeuristicDeducer(ExpressionGenerator(names), mechanics)),
        )
