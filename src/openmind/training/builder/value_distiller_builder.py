from typing import Self

from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.inference.service.deduction_inducer import DeductionInducer
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.service.position_ponderer import PositionPonderer
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.value_distiller import ValueDistiller
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class ValueDistillerBuilder:
    """Sets how many worker processes a value distiller's self-play games, deductions and term evaluations run in, 1 by
    default, and wires the services it works with."""

    def __init__(self) -> None:
        self._workers = 1

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def build(self) -> ValueDistiller:
        state_reader, solver, predictor = StateReader(), SolverBuilder().build(), PredictorBuilder().build()
        rule_compiler, rule_runner = RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper()))
        consequence_library = ConsequenceLibraryBuilder().build()
        generator = ExpressionGenerator(VariableNameMapper())
        ponderer = PositionPonderer(
            PositionDeducer(solver, predictor, state_reader, ActionTextMapper()),
            DeductionInducer(generator, VariableNameMapper()),
            generator,
            solver,
            rule_compiler,
            rule_runner,
            consequence_library,
            TaskRunner(self._workers),
        )
        return ValueDistiller(
            SelfPlay(solver, predictor, state_reader, TaskRunner(self._workers)),
            ValueGeneratorBuilder().with_workers(self._workers).build(),
            PositionRowMapper(state_reader),
            rule_compiler,
            rule_runner,
            consequence_library,
            ponderer,
        )
