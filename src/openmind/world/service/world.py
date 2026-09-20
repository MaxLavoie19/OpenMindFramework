import logging
import threading

from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class World:
    """The state as it stands now, for whoever acts and whoever plans.

    An integrator pushes what it perceived and OMF keeps it here; the actor and the planner read the same state, so the
    actor answers what is actually happening and the planner prunes what can no longer be reached. Reading and
    replacing are guarded, since the two run in different threads.

    What arrives is decoded into the state's models. Until the codec step, an integrator pushes a state."""

    def __init__(self, state: State) -> None:
        self._state = state
        self._lock = threading.Lock()
        self._changes = 0

    def current(self) -> State:
        """The state as it stands."""
        with self._lock:
            return self._state

    def perceived(self, state: State) -> State:
        """Takes what was perceived as the state now, and gives it back."""
        with self._lock:
            if state != self._state:
                self._changes += 1
                self._state = state
            return self._state

    def happened(self, action: Action, outcome: State) -> State:
        """Takes what an action left as the state now: what the integrator says came of it, not what OMF predicted."""
        logger.debug("%s happened", action.name)
        return self.perceived(outcome)

    def changes(self) -> int:
        """How many times the state has changed: what tells a planner its tree has moved on."""
        with self._lock:
            return self._changes
