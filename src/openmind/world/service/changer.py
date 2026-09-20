import logging
from collections.abc import Sequence

from openmind.structure.model.grid import Grid
from openmind.structure.model.scalar import Scalar
from openmind.world.model.change import Change, Moved, Placed, Removed, Told
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Changer:
    """Makes the changes an action says it makes.

    An effect that hands back a position leaves what the action did to be worked out from the difference, and some
    of it cannot be worked out at all. Said as changes and made here, the position is what the changes leave and
    the changes are there to be read."""

    def applied(self, state: State, changes: Sequence[Change]) -> State:
        """The position those changes leave, made in the order they are given.

        The order is the whole of it where two changes touch the same square: what stands there is removed and then
        something moves onto it, which takes a piece, rather than something moving onto it and then being removed
        itself."""
        for change in changes:
            state = self._made(state, change)
        return state

    def between(self, before: State, after: State) -> tuple[Change, ...]:
        """The changes that took one position to the other, read off the two of them.

        An integrator who has declared nothing still answers what happens: the action is played and a position
        comes back. What the action did is then the difference — a cell that emptied is something removed, a cell
        that filled is something put down, and a value gone from one cell and arrived at another is that same thing
        carried. A piece taken in passing falls out as a removal naming a square the move never names, which is the
        fact that cannot be had from the two squares of the action.

        Pairing what left one cell with what arrived at another is the one thing inferred rather than read. A cell
        that received something did not send that same thing away, so it is never offered as where the thing came
        from — otherwise a rook taking a rook pairs the taken one with the square it stands on and reads as a piece
        moving nowhere. Where two alike things move at once it can still pair them the wrong way round, and a game
        able to say what it did should say it; this is for the game that says nothing.

        Removals come first, then what was carried, then what was put down — so a piece is taken before something
        moves onto its square rather than after."""
        removed: list[Change] = []
        moved: list[Change] = []
        placed: list[Change] = []
        for name, model in after.models:
            if not before.has(name):
                continue
            was = before.model(name)
            if isinstance(model, Grid) and isinstance(was, Grid):
                one, other, carried = self._cells(name, was, model)
                removed.extend(one)
                placed.extend(other)
                moved.extend(carried)
            elif isinstance(model, Scalar) and isinstance(was, Scalar) and model.value != was.value:
                placed.append(Told(name, model.value))
        return (*removed, *moved, *placed)

    def _cells(self, name: str, was: Grid, now: Grid) -> tuple[list[Change], list[Change], list[Change]]:
        """What emptied, what filled, and what of it was one thing carried from here to there."""
        emptied = [(at, was.at(at)) for at in was.coordinates() if was.at(at) is not None and now.at(at) != was.at(at)]
        filled = [(at, now.at(at)) for at in now.coordinates() if now.at(at) is not None and now.at(at) != was.at(at)]
        arrived = {at for at, _ in filled}
        carried: list[Change] = []
        for at, value in list(filled):
            source = next((one for one, held in emptied if held == value and one not in arrived), None)
            if source is None:
                continue
            carried.append(Moved(name, source, at))
            emptied.remove((source, value))
            filled.remove((at, value))
        return (
            [Removed(name, at) for at, _ in emptied],
            [Placed(name, at, value) for at, value in filled],
            carried,
        )

    def _made(self, state: State, change: Change) -> State:
        if isinstance(change, Told):
            return state.with_model(change.model, change.value)
        grid = state.model(change.model)
        if not isinstance(grid, Grid):
            raise TypeError(f"{change.model!r} holds no cells to change")
        if isinstance(change, Placed):
            return state.with_model(change.model, grid.placed(change.at, change.value))
        if isinstance(change, Removed):
            return state.with_model(change.model, grid.removed(change.at))
        if isinstance(change, Moved):
            return state.with_model(change.model, grid.moved(change.source, change.target))
        raise TypeError(f"Unknown change: {change!r}")
