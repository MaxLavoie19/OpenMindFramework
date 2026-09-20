from openmind.inference.service.heuristic_ponderer import HeuristicPonderer
from openmind.inference.service.heuristic_proposer import HeuristicProposer
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.model.factory.model_factory import create_model_registry
from openmind.rbs.factory.rbs_factory import create_value_generator
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.service.state_reader import StateReader


def create_position_deducer() -> PositionDeducer:
    """Reasoning about one position with nothing but the game's own rules."""
    return PositionDeducer(StateReader(), ActionTextMapper())


def create_heuristic_ponderer(knowledge_base: KnowledgeBase, workers: int = 1) -> HeuristicPonderer:
    """The ponderer that deduces a game's first heuristics from its rules, with its gatherer, its deducer, its value
    generator and the relaxer it tries a game's relaxations with."""
    return HeuristicPonderer(
        PositionGatherer(), create_position_deducer(), create_value_generator(workers), GameRelaxer(knowledge_base)
    )


def create_heuristic_proposer() -> HeuristicProposer:
    """Proposing heuristics from a game's own vocabulary, before it has been played."""
    return HeuristicProposer(ExpressionGenerator(), create_model_registry())
