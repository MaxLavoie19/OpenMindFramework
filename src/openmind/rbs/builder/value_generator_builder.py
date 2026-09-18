from typing import Self

from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_search import ExpressionSearch
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner
from openmind.rbs.service.sparse_fitter import SparseFitter
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class ValueGeneratorBuilder:
    """Sets how many worker processes a value generator evaluates expressions in, 1 by default, and the memory each of
    them holds at most, no cap by default, and wires it: one expression generator and one sparse fitter shared by its
    expression search and itself."""

    def __init__(self) -> None:
        self._workers = 1
        self._memory_cap: MemoryCap | None = None

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def with_memory_cap(self, memory_cap: MemoryCap | None) -> Self:
        self._memory_cap = memory_cap
        return self

    def build(self) -> ValueGenerator:
        names = VariableNameMapper()
        compiler, runner, library = RuleCompiler(), RuleRunner(StateNamespaceMapper(names)), ConsequenceLibraryBuilder().build()
        evaluator = TermEvaluator(
            compiler,
            runner,
            library,
            TaskRunner(self._workers, self._memory_cap),
            ReadingCache(compiler, runner, library, MemoryMeter()),
        )
        generator, fitter = ExpressionGenerator(names), SparseFitter()
        search = ExpressionSearch(generator, evaluator, fitter, MemoryMeter())
        return ValueGenerator(search, generator, fitter)
