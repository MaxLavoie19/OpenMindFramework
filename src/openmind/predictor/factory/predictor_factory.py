from openmind.predictor.service.effects_runner import EffectsRunner
from openmind.predictor.service.rule_predictor import RulePredictor
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.world.mapper.action_text_mapper import ActionTextMapper


def create_effects_runner() -> EffectsRunner:
    """An effects runner with its own rule caller."""
    return EffectsRunner(create_rule_caller(), ActionTextMapper())


def create_rule_predictor() -> RulePredictor:
    """The predictor model running an RBS's effects rules."""
    return RulePredictor(create_effects_runner())
