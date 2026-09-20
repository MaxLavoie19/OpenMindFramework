from typing import Protocol

from openmind.world.model.action import Action


class Dispatcher(Protocol):
    """What performs the actions a strategy calls for.

    OMF's own runs a thread of its own; an integrator fills this port with whatever performs its actions — a robot's
    control loop, a game client's connection, an API. Several synchronous actions can be performed at once, such as
    motors moving into position, and an asynchronous one doesn't hold the rest up.

    `performing` is what has been dispatched and isn't done, which is what tells the agent whether it is still acting."""

    def dispatch(self, action: Action) -> None: ...

    def performing(self) -> tuple[Action, ...]: ...
