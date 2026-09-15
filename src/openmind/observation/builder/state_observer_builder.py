from openmind.observation.service.state_observer import StateObserver
from openmind.rule.factory.rule_factory import create_rule_caller


class StateObserverBuilder:
    """Wires a state observer with its rule compiler and rule runner."""

    def build(self) -> StateObserver:
        return StateObserver(create_rule_caller())
