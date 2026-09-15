from openmind.observation.builder.state_observer_builder import StateObserverBuilder
from openmind.observation.service.state_observer import StateObserver


def create_state_observer() -> StateObserver:
    """A state observer running observation rules with its own rule compiler and runner."""
    return StateObserverBuilder().build()
