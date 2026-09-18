from openmind.inference.mapper.expression_sentence_mapper import ExpressionSentenceMapper
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.csp.factory.csp_factory import create_solver
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rbs.repository.explanation_cache_repository import ExplanationCacheRepository
from openmind.rbs.service.rule_explainer import RuleExplainer
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


def create_value_generator(workers: int = 1) -> ValueGenerator:
    """A value generator with its term generator, term evaluator and sparse fitter, evaluating terms in that many worker
    processes."""
    return ValueGeneratorBuilder().with_workers(workers).build()


def create_rule_explainer() -> RuleExplainer:
    """A rule explainer reading rules literally, asking a language model when given one, and caching its sentences."""
    return RuleExplainer(
        ExpressionSentenceMapper(), ExpressionGenerator(VariableNameMapper()), ExplanationCacheRepository()
    )


def create_rule_based_system(knowledge_base: KnowledgeBase, context: str) -> RuleBasedSystem:
    """The RBS for a context: the rules the knowledge base holds for it, with the CSP as its solver for legal moves and
    a predictor for what a move leads to. A context the knowledge base holds no rule for gives an RBS that can't say
    where the game starts."""
    return RuleBasedSystem(
        context,
        knowledge_base.rules(context),
        create_solver(),
        create_predictor(),
        create_rule_caller(),
        StateReader(),
        ConsequenceLibraryBuilder().build(),
    )
