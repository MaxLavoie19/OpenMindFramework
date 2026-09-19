from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.csp.factory.csp_factory import create_solver
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.world.service.state_reader import StateReader


def create_value_generator(workers: int = 1) -> ValueGenerator:
    """A value generator with its term generator, term evaluator and sparse fitter, evaluating terms in that many worker
    processes."""
    return ValueGeneratorBuilder().with_workers(workers).build()


def create_rule_based_system(knowledge_base: KnowledgeBase, context: str) -> RuleBasedSystem:
    """The RBS for a context, named as applications name it: the rules the knowledge base holds for it, with the CSP as its solver for legal moves and
    a predictor for what a move leads to. A context the knowledge base holds no rule for gives an RBS that can't say
    where the game starts."""
    known = knowledge_base.context_named(context)
    return RuleBasedSystem(
        context,
        () if known is None else knowledge_base.rules(known.id),
        create_solver(),
        create_predictor(),
        create_rule_caller(),
        StateReader(),
        ConsequenceLibraryBuilder().build(),
        None if known is None else known.id,
    )
