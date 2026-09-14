from typing import Self

from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.value_rule_text_mapper import ValueRuleTextMapper
from openmind.rbs.service.sparse_fitter import SparseFitter
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rbs.service.term_generator import TermGenerator
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class ValueGeneratorBuilder:
    """Sets how many worker processes a value generator evaluates terms in, 1 by default, and wires it: one term
    evaluator shared by its term generator and itself."""

    def __init__(self) -> None:
        self._workers = 1

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def build(self) -> ValueGenerator:
        names = VariableNameMapper()
        state_namespace_mapper = StateNamespaceMapper(names)
        evaluator = TermEvaluator(
            RuleCompiler(), RuleRunner(state_namespace_mapper), ConsequenceLibraryBuilder().build(), TaskRunner(self._workers)
        )
        return ValueGenerator(
            TermGenerator(evaluator, state_namespace_mapper, names), evaluator, SparseFitter(), ValueRuleTextMapper()
        )
