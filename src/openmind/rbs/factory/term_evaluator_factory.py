from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner


def create_term_evaluator(workers: int = 1, memory_cap: MemoryCap | None = None) -> TermEvaluator:
    """What reads a term over a set of positions, with the compiler, runner, library, task runner and cache it
    needs.

    A module of its own because two things want one and one of them builds the other: the search that fits
    terms is built by `HeuristicFinderBuilder`, which `rbs_factory` imports, so a function the builder calls
    cannot live in that factory. Five constructor arguments are not a thing to write out twice either, which
    is how the two would otherwise drift apart."""
    compiler, runner, library = RuleCompiler(), RuleRunner(StateNamespaceMapper()), ConsequenceLibraryBuilder().build()
    return TermEvaluator(
        compiler,
        runner,
        library,
        TaskRunner(workers, memory_cap),
        ReadingCache(compiler, runner, library, MemoryMeter()),
    )
