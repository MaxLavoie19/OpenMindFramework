from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.model.service.factor_weights import FactorWeights
from openmind.model.service.model_drawer import ModelDrawer
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


def create_factor_weights() -> FactorWeights:
    """What each way of making a heuristic has been worth, so the next is made the way that has paid."""
    return FactorWeights()
