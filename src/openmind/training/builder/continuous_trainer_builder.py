from typing import Self

from openmind.agent.service.game_memory import GameMemory
from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.inference.service.deduction_inducer import DeductionInducer
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.mechanics import Mechanics
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.candidate_signals import CandidateSignals
from openmind.training.service.continuous_trainer import ContinuousTrainer
from openmind.training.service.game_replayer import GameReplayer
from openmind.training.service.game_study import GameStudy
from openmind.training.service.heuristic_deducer import HeuristicDeducer
from openmind.training.service.lesson_learner import LessonLearner
from openmind.training.service.online_value_fitter import OnlineValueFitter
from openmind.training.service.position_ponderer import PositionPonderer
from openmind.training.service.rule_searcher import RuleSearcher
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.signal_library_updater import SignalLibraryUpdater
from openmind.training.service.signal_preparer import SignalPreparer
from openmind.training.service.signal_ranker import SignalRanker
from openmind.training.service.signal_recorder import SignalRecorder
from openmind.training.service.signal_targeter import SignalTargeter
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class ContinuousTrainerBuilder:
    """Sets how many worker processes continuous training plays and studies its games in, and its rule search evaluates
    terms in, 1 by default; the memory each of them holds at most, no cap by default; and the knowledge base every game,
    proof and seed is remembered in, which it needs. A game's study runs inside its worker, on services of its own."""

    def __init__(self) -> None:
        self._workers = 1
        self._memory_cap: MemoryCap | None = None
        self._knowledge_base: KnowledgeBase | None = None

    def with_workers(self, workers: int) -> Self:
        self._workers = workers
        return self

    def with_memory_cap(self, memory_cap: MemoryCap | None) -> Self:
        self._memory_cap = memory_cap
        return self

    def with_knowledge_base(self, knowledge_base: KnowledgeBase) -> Self:
        self._knowledge_base = knowledge_base
        return self

    def build(self) -> ContinuousTrainer:
        if self._knowledge_base is None:
            raise ValueError("Continuous training needs a knowledge base to remember its games in")
        names = VariableNameMapper()
        state_reader, solver, predictor = StateReader(), SolverBuilder().build(), PredictorBuilder().build()
        rule_compiler, rule_runner = RuleCompiler(), RuleRunner(StateNamespaceMapper(names))
        consequence_library = ConsequenceLibraryBuilder().build()
        generator = ExpressionGenerator(names)
        here = TaskRunner(1)
        study_terms = TermEvaluator(
            rule_compiler, rule_runner, consequence_library, here, ReadingCache(rule_compiler, rule_runner, consequence_library, MemoryMeter())
        )
        study = GameStudy(
            SelfPlay(solver, predictor, state_reader, here),
            PositionPonderer(
                PositionDeducer(solver, predictor, state_reader, ActionTextMapper()),
                DeductionInducer(generator, names),
                generator,
                solver,
                rule_compiler,
                rule_runner,
                consequence_library,
                here,
            ),
            SignalRecorder(study_terms),
            SignalTargeter(study_terms),
            SignalRanker(),
            CandidateSignals(generator),
            study_terms,
            PositionRowMapper(state_reader),
        )
        runner = TaskRunner(self._workers, self._memory_cap)
        search_terms = TermEvaluator(
            rule_compiler, rule_runner, consequence_library, runner, ReadingCache(rule_compiler, rule_runner, consequence_library, MemoryMeter())
        )
        mechanics = Mechanics(solver, predictor, StateNamespaceMapper(names), MemoryMeter())
        return ContinuousTrainer(
            study,
            LessonLearner(SignalRecorder(search_terms), OnlineValueFitter(), self._knowledge_base),
            RuleSearcher(
                SignalTargeter(search_terms),
                SignalRanker(),
                ValueGeneratorBuilder().with_workers(self._workers).with_memory_cap(self._memory_cap).build(),
                SignalLibraryUpdater(),
            ),
            GameMemory(self._knowledge_base),
            GameReplayer(predictor),
            SignalPreparer(HeuristicDeducer(ExpressionGenerator(names), mechanics)),
            SelfPlay(solver, predictor, state_reader, here),
            ArmSelector(),
            runner,
            rule_compiler,
            rule_runner,
            consequence_library,
        )
