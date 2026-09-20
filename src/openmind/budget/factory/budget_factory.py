from openmind.budget.service.plain_time_manager import PlainTimeManager
from openmind.model.factory.model_factory import create_model_registry


def create_plain_time_manager() -> PlainTimeManager:
    """The bootstrap time management model, reading what the model registry has measured."""
    return PlainTimeManager(create_model_registry())
