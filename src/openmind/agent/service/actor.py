import logging
import threading

from openmind.agent.model.dispatcher import Dispatcher
from openmind.search.model.strategy import Strategy
from openmind.world.model.state import State
from openmind.world.service.world import World

logger = logging.getLogger(__name__)

#: How long the actor waits before looking at the world again, in seconds.
LOOK_AGAIN_SECONDS = 0.01


class Actor:
    """Performs what a strategy calls for, in a thread of its own, while the planner keeps strategizing.

    It reads the world as it stands and dispatches what the strategy says to play there. Where the strategy has nothing
    prepared for the state it is in, it waits: that is the surprise case, and the planner is already strategizing from
    there. It acts once per state, so a state it has acted in isn't acted in twice.

    It opens no debugger frames: a pause in this thread would hold the game up, and what it did is in the logs."""

    def __init__(self, dispatcher: Dispatcher, wait_seconds: float = LOOK_AGAIN_SECONDS) -> None:
        self._dispatcher = dispatcher
        self._wait = wait_seconds
        self._strategy: Strategy | None = None
        self._acted: State | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def follow(self, strategy: Strategy | None) -> None:
        """The strategy to act on from now on: what the planner has worked out so far."""
        self._strategy = strategy

    def act(self, world: World) -> bool:
        """Dispatches what the strategy says to play in the state as it stands; whether it acted. Without a strategy,
        or with one that says nothing here, it acts not at all."""
        strategy, state = self._strategy, world.current()
        if strategy is None or state == self._acted:
            return False
        action = strategy.chosen(state)
        if action is None:
            logger.debug("Nothing prepared for this state: waiting while the planner strategizes")
            return False
        self._acted = state
        self._dispatcher.dispatch(action)
        logger.info("Dispatched %s", action.name)
        return True

    def start(self, world: World) -> None:
        """Acts on the world in a thread of its own until stopped."""
        if self._thread is not None:
            raise ValueError("The actor is already acting")
        self._stop.clear()
        self._thread = threading.Thread(target=self._acting, args=(world,), name="actor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the thread and waits for it to end; acting again needs another start."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()

    def acting(self) -> bool:
        """Whether its thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _acting(self, world: World) -> None:
        while not self._stop.is_set():
            self.act(world)
            self._stop.wait(self._wait)
