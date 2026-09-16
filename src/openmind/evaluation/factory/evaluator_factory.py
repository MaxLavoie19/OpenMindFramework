from openmind.agent.service.game_memory import GameMemory
from openmind.evaluation.builder.evaluator_builder import EvaluatorBuilder
from openmind.evaluation.service.evaluator import Evaluator


def create_evaluator(workers: int = 1, game_memory: GameMemory | None = None) -> Evaluator:
    """An evaluator with its match runner, exact and reference searches, running games and searches in that many
    worker processes, and remembering every match game in the game memory when given."""
    return EvaluatorBuilder().with_workers(workers).with_game_memory(game_memory).build()
