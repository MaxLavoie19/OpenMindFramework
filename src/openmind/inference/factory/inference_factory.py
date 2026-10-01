from openmind.inference.service.choice_fitter import ChoiceFitter
from openmind.inference.service.heuristic_ponderer import HeuristicPonderer
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.inference.service.stability import Stability
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.rbs.factory.rbs_factory import create_heuristic_finder
from openmind.rbs.factory.term_evaluator_factory import create_term_evaluator
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.service.state_reader import StateReader


def create_position_deducer() -> PositionDeducer:
    """Reasoning about one position with nothing but the game's own rules."""
    return PositionDeducer(StateReader(), ActionTextMapper())


def create_choice_fitter() -> ChoiceFitter:
    """What fits a preference to what somebody chose, where nobody said what anything was worth. Stateless,
    so it takes nothing."""
    return ChoiceFitter()


def create_heuristic_ponderer(
    knowledge_base: KnowledgeBase, workers: int = 1, allowance: float = 0.0, telling=None
) -> HeuristicPonderer:
    """The ponderer that deduces a game's first heuristics from its rules, with its gatherer, its deducer, its value
    generator, the relaxer it tries a game's relaxations with, the three services that reason out what the
    things in a game are worth and write that as terms for the search to start from, and what measures how
    steadily a term reads so the search reaches the steady ones sooner.

    With an allowance above nought its finder runs an economy: a fitted term is written into a ruleset only
    where some signal staked a budget on it. At nought every term the fit keeps is written.

    `telling` is how to ask somebody who knows what each position is worth, where the domain has somebody to
    ask. Given none the rows carry nothing but what the games paid, and the signal that reads a teller wants
    nothing — which is what having nobody to ask looks like."""
    expression_generator = ExpressionGenerator()
    return HeuristicPonderer(
        PositionGatherer(),
        create_position_deducer(),
        create_heuristic_finder(workers, allowance),
        GameRelaxer(knowledge_base),
        expression_generator,
        create_term_evaluator(workers),
        Stability(),
        telling,
    )

