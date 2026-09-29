from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.model.service.model_drawer import ModelDrawer
from openmind.model.service.rule_budget import RuleBudget
from openmind.model.service.rule_price import RulePrice
from openmind.model.service.model_registry import ModelRegistry
from openmind.model.service.model_timer import ModelTimer


def create_model_registry() -> ModelRegistry:
    """The model registry, with the accuracy scorer measuring the mechanisms its models read through."""
    return ModelRegistry(AccuracyScorer())


def create_model_drawer() -> ModelDrawer:
    """The model drawer, over the registry that holds what fills each task.

    Its own registry rather than one shared with a caller: a registry keeps nothing of its own, taking the
    knowledge base on every call, so two of them are the same thing twice and neither can drift."""
    return ModelDrawer(create_model_registry())


def create_model_timer() -> ModelTimer:
    """The model timer, on wall time."""
    return ModelTimer()



def create_rule_budget() -> RuleBudget:
    """What each signal has earned and spent vouching for rules."""
    return RuleBudget()


def create_rule_price() -> RulePrice:
    """What a rule costs to vouch for, in the bits it takes to write down."""
    return RulePrice()
