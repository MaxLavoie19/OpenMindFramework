from collections.abc import Callable

from openmind.agent.service.outfitter import Outfitter
from openmind.heuristic.service.rule_position_valuer import RulePositionValuer
from openmind.heuristic.service.timed_position_valuer import TimedPositionValuer
from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.service.model_registry import ModelRegistry
from openmind.model.service.model_timer import ModelTimer
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_rule_heuristic
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule

type Game = Callable[[str], RuleBasedGame]


def a_model(knowledge: KnowledgeBase, game: Game, heuristic: Callable[..., object]) -> ModelRecord:
    """A game with a position value heuristic, and the model the registry holds for it."""
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 1.0)
    context = knowledge.context_named("tictactoe")
    return ModelRegistry(AccuracyScorer()).of_task(knowledge, context.id, POSITION_VALUE)[0]


def test_every_reading_of_a_chosen_model_is_timed(
    knowledge: KnowledgeBase, game: Game, heuristic: Callable[..., object]
) -> None:
    """What a planner may explore is worked out from what its models have been costing, and nothing was
    measuring what they cost — so every model cost the unmeasured default of a millisecond, and against chess
    that was out by three hundred times: two seconds of intended thinking bought ten minutes of actual
    thinking and no game ever finished."""
    record = a_model(knowledge, game, heuristic)

    filled = Outfitter(create_rule_heuristic(), ModelTimer()).filled(knowledge, record)

    assert isinstance(filled[1], TimedPositionValuer)


def test_what_a_reading_cost_is_kept_where_the_time_manager_looks_for_it(
    knowledge: KnowledgeBase, game: Game, heuristic: Callable[..., object]
) -> None:
    """Both halves existed and agreed — `ModelTimer` writes what `PlainTimeManager` reads — and nothing ever
    called the timer. This is the loop closed: read once, and the registry knows what it cost."""
    record = a_model(knowledge, game, heuristic)
    model, valuer = Outfitter(create_rule_heuristic(), ModelTimer()).filled(knowledge, record)
    played = create_rule_based_game(knowledge, "tictactoe")

    valuer.values(model, played.node(played.start()))

    measured = ModelRegistry(AccuracyScorer()).measured(knowledge, record)
    assert measured.readings == 1
    assert measured.processing_seconds is not None and measured.processing_seconds >= 0.0


def test_without_a_timer_the_service_is_handed_on_as_it_is(
    knowledge: KnowledgeBase, game: Game, heuristic: Callable[..., object]
) -> None:
    """Timing is given rather than assumed, so a caller that does not want its readings measured does not pay
    for measuring them."""
    record = a_model(knowledge, game, heuristic)

    filled = Outfitter(create_rule_heuristic()).filled(knowledge, record)

    assert isinstance(filled[1], RulePositionValuer)
