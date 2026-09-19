from openmind.epistemology.service.accuracy_scorer import AccuracyScorer
from openmind.model.service.model_registry import ModelRegistry
from openmind.model.service.model_timer import ModelTimer


def create_model_registry() -> ModelRegistry:
    """The model registry, with the accuracy scorer measuring the mechanisms its models read through."""
    return ModelRegistry(AccuracyScorer())


def create_model_timer() -> ModelTimer:
    """The model timer, on wall time."""
    return ModelTimer()
