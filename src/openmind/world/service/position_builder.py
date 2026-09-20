import logging
import random
from collections.abc import Mapping, Sequence

from openmind.structure.model.grid import Grid
from openmind.structure.model.value import Value
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class PositionBuilder:
    """Positions built on purpose, rather than reached by playing.

    What a game shows of itself by being played is a narrow thing: chess's opening has every piece blocked by its
    own, and no walk from it ever reaches a board with one bishop on it. A position built on purpose does — and a
    position built on purpose is how a rule is put to the test, since the way to find out whether a rule holds is to
    build the position that would break it and ask.

    It takes a position and gives another: everything cleared, or some cells set to something. Whether what comes out
    is a position the game could have reached is not asked — an impossible position still answers what the rules
    allow in it, which is what is being found out."""

    def emptied(self, state: State) -> State:
        """The position with every grid emptied: every cell holding nothing, and everything else as it was."""
        emptied = state
        for name, model in state.models:
            if isinstance(model, Grid):
                emptied = emptied.with_model(name, Grid.filled(model.shape, None, model.aliases, model.directions))
        return emptied

    def placed(self, state: State, placements: Mapping[str, Mapping[object, Value]]) -> State:
        """The position with those cells set: `{"piece": {"e4": "queen"}, "color": {"e4": "white"}}`."""
        placed = state
        for name, cells in placements.items():
            grid = placed.model(name)
            for where, value in cells.items():
                grid = grid.placed(where, value)  # type: ignore[union-attr]
            placed = placed.with_model(name, grid)
        return placed

    def scattered(
        self,
        state: State,
        values: Mapping[str, Sequence[Value]],
        pieces: int,
        rng: random.Random,
        scalars: Mapping[str, Sequence[Value]] | None = None,
    ) -> State:
        """An emptied position with that many cells filled at random, each grid taking one of the values it was given
        for that cell — a board with a handful of things on it, and nothing like a game.

        The same cells are filled in every grid, so what a game keeps in one grid stays with what it keeps in
        another: a piece and whose it is.

        `scalars` says what the position's own scalars may be. Leaving them as they were is how a built position
        quietly stays like the one it came from — every board with the same player to move — and a rule that only
        held because of that survives a test it should have failed."""
        built = self.emptied(state)
        grids = [name for name, model in built.models if isinstance(model, Grid) and name in values]
        if not grids:
            return built
        cells = list(built.model(grids[0]).coordinates())  # type: ignore[union-attr]
        chosen = rng.sample(cells, min(pieces, len(cells)))
        built = self.placed(built, {name: {cell: rng.choice(values[name]) for cell in chosen} for name in grids})
        for name, held in (scalars or {}).items():
            built = built.with_model(name, rng.choice(list(held)))
        return built
