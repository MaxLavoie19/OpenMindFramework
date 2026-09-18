from openmind.predictor.service.predictor import Predictor
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.world.mapper.action_text_mapper import ActionTextMapper


class PredictorBuilder:
    """Wires a predictor with its rule compiler, rule runner and action text mapper."""

    def build(self) -> Predictor:
        return Predictor(create_rule_caller(), ActionTextMapper())
