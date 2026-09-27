from dataclasses import dataclass, fields

from openmind.structure.model.grid import Grid
from openmind.structure.model.record import Record
from openmind.structure.model.value import Value
from openmind.statement.model.change import Change, Moved, Placed, Removed, Told
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Happening:
    """What can come about: one or more changes, together.

    The thing a notation names and constraints are over. A plain move is one change; a castling is two; a
    promotion is a move and a placement. Said this way they stop being special cases — a happening with more
    changes in it, rather than an action with a stranger shape."""

    changes: tuple[Change, ...]

    def said(self, state: State) -> dict[str, Value]:
        """What this happening is, in parts something could name.

        **Every part is read off the changes and the position, and none is declared.** Where a thing is carried
        from and to are the change's own places; what the thing *is* comes from whatever stands there, opened up
        by its own parts — so a game whose pieces have a colour and a kind offers both, and a game whose cells
        hold a number offers that. Whether anything was taken is whether a removal is among the changes.

        These are what a notation can be found to talk about. Nothing here knows that a notation exists."""
        found: dict[str, Value] = {"takes": "yes" if any(isinstance(one, Removed) for one in self.changes) else "no"}
        found["changes"] = len(self.changes)
        for change in self.changes:
            for name, where in self._places(change):
                for number, one in enumerate(where, start=1):
                    found[f"{name} place {number}"] = one
                for part, value in self._parts(state, change.model, where):
                    found[f"{name} {part}"] = value
            if isinstance(change, Told):
                found[f"told {change.model}"] = change.value
        return found

    def _places(self, change: Change) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """A change's places, by the names the change gives them."""
        if isinstance(change, Moved):
            return (("from", change.source), ("to", change.target))
        if isinstance(change, (Placed, Removed)):
            return (("at", change.at),)
        return ()

    def _parts(self, state: State, model: str, where: tuple[int, ...]) -> tuple[tuple[str, Value], ...]:
        """What stands at that place before the happening, opened up by its own parts.

        A thing with named parts offers each of them; a thing without offers itself. This is what lets a symbol
        be found to name a kind rather than a whole piece."""
        if not state.has(model):
            return ()
        held = state.model(model)
        if not isinstance(held, Grid) or not held.inside(where):
            return ()
        standing = held.at(where)
        if isinstance(standing, Record):
            return tuple((one.name, getattr(standing, one.name)) for one in fields(standing))
        return (("holds", standing),)
