import logging
import re
from collections.abc import Mapping, Sequence

from openmind.inference.model.sides import Sides
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: How the reading naming the step between an action's two cells is spelled, and what it names them.
STEP = re.compile(r"^rows from (?P<first>.+) to (?P<second>.+)$")


class SideDeducer:
    """Works out what belongs to whom, and which way each player faces, from the actions a game allows.

    A player's own side is the one thing about a board game that is nowhere on the board. It can be assumed — the
    first player faces up, the second faces down — but that is false of a game where both play the same way up, and
    an assumption that is false of some games is what OMF is not allowed to hold. It can be declared, but the whole
    exercise is that the integrator declared nothing.

    So it is read off what the game allows. A value is a player's where that player's actions move it: whatever
    stands where a move begins belongs to whoever may play it. And a player faces the way their one-way pieces go —
    a value of theirs that changes the row in one direction and never the other, in position after position, is a
    piece that advances, and a piece that advances says which way forward is. One such value orients the whole side;
    a game that has none has no side to tell, and then nothing is deduced and nothing is offered."""

    def deduce(
        self,
        examples: Sequence[tuple[Mapping[str, Value], Value]],
        positions: Sequence[object] = (),
        least: int = 2,
    ) -> Sides:
        """From legal actions and whose they are: what belongs to whom, and which way each player faces.

        `examples` are the readings of actions the game allows, each with the player who may play it. `positions`
        says which position each was read in, and `least` how many positions a one-way value must be seen going one
        way in — seen in a single position, a rook that happened to move up twice is as one-way as a pawn."""
        where = list(positions) if len(positions) == len(examples) else [None] * len(examples)
        owning = self._owning(examples)
        facing = self._facing(examples, owning, where, least)
        logger.info("Deduced %d values as a player's own, and which way %d players face", len(owning), len(facing))
        return Sides(owning, facing)

    def _owning(self, examples: Sequence[tuple[Mapping[str, Value], Value]]) -> tuple[tuple[str, Value, Value], ...]:
        """What belongs to whom: a value only ever moved by one player is that player's.

        A value several players move belongs to none of them — a neutral piece either side may push — and saying so
        is better than handing it to whoever moved it more often."""
        movers: dict[tuple[str, Value], set[Value]] = {}
        for readings, player in examples:
            for model, value in self._moved(readings):
                movers.setdefault((model, value), set()).add(player)
        return tuple(
            sorted(
                ((model, value, next(iter(who))) for (model, value), who in movers.items() if len(who) == 1),
                key=repr,
            )
        )

    def _facing(
        self,
        examples: Sequence[tuple[Mapping[str, Value], Value]],
        owning: tuple[tuple[str, Value, Value], ...],
        where: Sequence[object],
        least: int,
    ) -> tuple[tuple[Value, int], ...]:
        """Which way each player faces: the way a thing of theirs goes, where it never goes the other way.

        What advances is a white pawn, and neither half of that says so: pawns are moved by both players, and the
        white things on the board include a rook that goes both ways. So a one-way thing is looked for in what
        stands where the move begins taken whole, and it is that player's where no other player ever moves it."""
        ways: dict[tuple[Value, tuple[tuple[str, Value], ...]], set[int]] = {}
        seen: dict[tuple[Value, tuple[tuple[str, Value], ...]], set[object]] = {}
        movers: dict[tuple[tuple[str, Value], ...], set[Value]] = {}
        for (readings, player), place in zip(examples, where, strict=True):
            holding = tuple(sorted(self._moved(readings), key=repr))
            if not holding:
                continue
            movers.setdefault(holding, set()).add(player)
            rows = self._rows(readings)
            if not rows:
                continue
            ways.setdefault((player, holding), set()).add((rows > 0) - (rows < 0))
            seen.setdefault((player, holding), set()).add(place)
        facing: dict[Value, set[int]] = {}
        for (player, holding), steps in ways.items():
            if len(steps) != 1 or len(seen[(player, holding)]) < least or movers[holding] != {player}:
                continue
            facing.setdefault(player, set()).update(steps)
            logger.info(
                "%s only ever moves %s for %r, so that is forward",
                " and ".join(f"{model} {value!r}" for model, value in holding),
                "up" if 1 in steps else "down",
                player,
            )
        return tuple(sorted(((player, steps.pop()) for player, steps in facing.items() if len(steps) == 1), key=repr))

    def _moved(self, readings: Mapping[str, Value]) -> list[tuple[str, Value]]:
        """What stands where the action begins, by the model it stands in."""
        begins = self._begins(readings)
        if not begins:
            return []
        ending = f" at {begins}"
        return [
            (name[: -len(ending)], value)
            for name, value in readings.items()
            if name.endswith(ending) and value is not None
        ]

    def _begins(self, readings: Mapping[str, Value]) -> str:
        """The parameter an action begins at: the first of the two cells a step is read between."""
        for name in readings:
            said = STEP.match(name)
            if said:
                return said.group("first")
        return ""

    def _rows(self, readings: Mapping[str, Value]) -> int:
        """How many rows the action moves, and nothing where it moves none or the action has no two cells."""
        for name, value in readings.items():
            if STEP.match(name) and isinstance(value, int) and not isinstance(value, bool):
                return value
        return 0

    def _whose(self, owning: tuple[tuple[str, Value, Value], ...], model: str, value: Value) -> Value | None:
        for held, one, player in owning:
            if held == model and one == value:
                return player
        return None
