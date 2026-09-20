from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Sides:
    """Which values on the board belong to which player, and which way each player faces.

    Neither is written anywhere in a position. A grid says a piece is white; nothing says white is the player to
    move, and nothing says white walks up the board. Without them, every rule about a piece of one's own is a rule
    about the colour white, learned again for black and saying nothing either time, and the rules that are about a
    player's own side — the rank a pawn starts on, the rank it promotes on — cannot be said at all.

    Both are facts about the game, so both are deduced from what the game allows rather than declared or assumed."""

    #: What belongs to whom: the model it stands in, the value, and the player whose it is.
    owning: tuple[tuple[str, Value, Value], ...] = ()

    #: Which way each player faces: the player, and the row step that is forward for them.
    facing: tuple[tuple[Value, int], ...] = ()

    def whose(self, model: str, value: Value) -> Value | None:
        """The player that value belongs to, or None where it belongs to nobody."""
        for held, one, player in self.owning:
            if held == model and one == value:
                return player
        return None

    def toward(self, player: Value) -> int:
        """The row step that is forward for that player, or 0 where the game has no side to tell."""
        for one, step in self.facing:
            if one == player:
                return step
        return 0
