from openmind.inference.service.heuristic_deriver import HeuristicDeriver
from openmind.inference.service.heuristic_ponderer import HeuristicPonderer
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.inference.service.stability import Stability
from openmind.inference.service.worth_reasoner import WorthReasoner
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.rbs.factory.rbs_factory import create_value_generator
from openmind.rbs.factory.term_evaluator_factory import create_term_evaluator
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.service.state_reader import StateReader


def create_position_deducer() -> PositionDeducer:
    """Reasoning about one position with nothing but the game's own rules."""
    return PositionDeducer(StateReader(), ActionTextMapper())


def create_heuristic_ponderer(knowledge_base: KnowledgeBase, workers: int = 1) -> HeuristicPonderer:
    """The ponderer that deduces a game's first heuristics from its rules, with its gatherer, its deducer, its value
    generator, the relaxer it tries a game's relaxations with, the three services that reason out what the
    things in a game are worth and write that as terms for the search to start from, and what measures how
    steadily a term reads so the search reaches the steady ones sooner."""
    expression_generator = ExpressionGenerator()
    return HeuristicPonderer(
        PositionGatherer(),
        create_position_deducer(),
        create_value_generator(workers),
        GameRelaxer(knowledge_base),
        HeuristicDeriver(expression_generator=expression_generator),
        WorthReasoner(),
        expression_generator,
        create_term_evaluator(workers),
        Stability(),
    )


def create_heuristic_deriver() -> HeuristicDeriver:
    """Reasoning a game's first heuristics out of its own rules, with the generator it writes them as terms with."""
    return HeuristicDeriver(expression_generator=ExpressionGenerator())
