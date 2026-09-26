from openmind.predictor.service.effects_runner import EffectsRunner
from openmind.predictor.service.rule_predictor import RulePredictor
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.mapper.action_text_mapper import ActionTextMapper


def create_effects_runner(rule_caller: RuleCaller | None = None) -> EffectsRunner:
    """An effects runner with its own rule caller, or with one it is given.

    **Given one, because what an action does is where a learned game needs its own.** A consequence may say
    *the player not acting*, and which of a position's models says who is acting is the game's to tell. A
    factory that builds its dependency cannot be told, so a game whose effects were learned played its moves
    through a caller that could not read its turn — and its turn never passed."""
    return EffectsRunner(create_rule_caller() if rule_caller is None else rule_caller, ActionTextMapper())


def create_rule_predictor(rule_caller: RuleCaller | None = None) -> RulePredictor:
    """The predictor model running an RBS's effects rules."""
    return RulePredictor(create_effects_runner(rule_caller))
