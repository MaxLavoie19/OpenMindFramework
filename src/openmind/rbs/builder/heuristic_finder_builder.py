from collections.abc import Sequence
from typing import Self

from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.expression_search import ExpressionSearch
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.model.model.rule_signal import RuleSignal
from openmind.model.service.rule_admission import RuleAdmission
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.factory.term_evaluator_factory import create_term_evaluator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.rbs.service.sparse_fitter import SparseFitter
from openmind.inference.service.term_assembler import TermAssembler
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rbs.service.heuristic_finder import HeuristicFinder


class HeuristicFinderBuilder:
    """Sets how many worker processes a heuristic finder evaluates expressions in, 1 by default, and the memory each of
    them holds at most, no cap by default, and wires it: one expression generator and one sparse fitter shared by its
    expression search and itself.

    It can also be given an economy: who has to vouch for a term before it is written into a ruleset, and with
    what budget. Without one, every term the fit keeps is written, which is what this did before there was an
    economy — an economy is something a caller turns on."""

    def __init__(self) -> None:
        self._workers = 1
        self._memory_cap: MemoryCap | None = None
        self._admission: RuleAdmission | None = None
        self._signals: tuple[RuleSignal, ...] = ()

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def with_memory_cap(self, memory_cap: MemoryCap | None) -> Self:
        self._memory_cap = memory_cap
        return self

    def with_admission(self, admission: RuleAdmission | None, signals: Sequence[RuleSignal] = ()) -> Self:
        self._admission, self._signals = admission, tuple(signals)
        return self

    def build(self) -> HeuristicFinder:
        evaluator = create_term_evaluator(self._workers, self._memory_cap)
        generator, fitter = ExpressionGenerator(), SparseFitter()
        search = ExpressionSearch(generator, evaluator, fitter, MemoryMeter())
        # The assembler reads the same rows the search does and builds what the search cannot grow to; the
        # evaluator it is given is the one that already times every reading, so an assembled term is priced
        # like any other.
        return HeuristicFinder(
            search, generator, fitter, self._admission, self._signals, TermAssembler(), evaluator
        )
