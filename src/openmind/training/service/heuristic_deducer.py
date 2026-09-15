import logging

from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import HERE, ME, OTHER
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.mechanics import Mechanics
from openmind.rbs.constant.consequence_constant import SOLO_DISTANCE
from openmind.rule.model.python_rule import PythonRule
from openmind.training.constant.signal_constant import (
    DEFAULT_GOAL_LIMIT,
    FORK,
    GOAL_DISTANCE,
    HANGING,
    LOSING,
    LOSING_MOVES,
    MATERIAL,
    MY_OPTIONS,
    OPTIONS,
    OWNED,
    TAKING,
    TAKING_MOVES,
    THEIR_OPTIONS,
)
from openmind.training.model.signal import Signal
from openmind.world.model.grid import Grid
from openmind.world.model.value import Value

logger = logging.getLogger(__name__)

#: A reading no taking move reaches: a look-ahead giving it found nothing to take.
NOTHING_TAKEN = "float('-inf')"


class HeuristicDeducer:
    """Deduces value signals from a two-player domain's rules alone, as general principles a clueless agent falls back
    on. No game has been played, so nothing is tested: games weigh them later, and trades worked out in play replace
    them. Every signal is normalized by the position's own totals, so it reads from -1 to 1, 0 when its divisor is 0.
    Each signal names its premises:

    - `options`, the player's legal moves minus the other player's, over both: more options is better; `my options`, the
      player's legal moves over both, and `their options`, minus the other player's over both: an opponent with fewer
      moves is easier;
    - `goal distance`, the other player's distance to a win minus the player's, over both, a distance being the fewest
      moves to a win if the other player did nothing: being closer to winning is better; blank when neither can win
      within the goal limit;
    - for every indexed base whose values in the initial position name players, a player's things: `owned <base>`, the
      player's things minus the other's, over both, since moves come from things; `taking <base>` and `losing <base>`,
      the most things the player can take with one move over the other's things, or minus the most the other player can
      take over the player's; `taking moves <base>` and `losing moves <base>`, the most legal moves the other player
      loses from one of the player's taking moves over their moves, or minus the most the player loses from one of the
      other's over the player's: things with many moves are worth more; `fork <base>`, how many of the player's moves
      leave at least two moves each taking one of the other's things, over the player's moves;
    - for a grid of such things, `material <base>`, the worth of the player's things minus the other's, over both, a
      thing worth the average legal moves of its kind alone on every cell, its kind being what the other grids with the
      same cells hold there (a piece in chess); and `hanging <base>`, minus how many of the player's things the other
      player can take where, were the thing the other player's, the player couldn't take it, over the player's things.

    The signals about taking, losing, forks and hanging things give None where there is nothing to take, lose, fork or
    leave hanging: they are blank there."""

    def __init__(self, expression_generator: ExpressionGenerator, mechanics: Mechanics) -> None:
        self._expression_generator = expression_generator
        self._mechanics = mechanics

    def deduce(self, domain: Domain, goal_limit: int = DEFAULT_GOAL_LIMIT) -> tuple[Signal, ...]:
        """The signals deduced, the options and goal distance first, then each base's; goal distance looks for a win up to
        `goal_limit` moves ahead. A domain without exactly two players raises ValueError."""
        players = domain.players.names
        if len(players) != 2:
            raise ValueError(f"Signals are deduced between two players, not {len(players)}")
        moves = f"(mine := {HERE}.mobility({ME})) + (theirs := {HERE}.mobility({OTHER}))"
        distances = (
            f"(mine := {SOLO_DISTANCE}({ME}, None, {goal_limit})) + (theirs := {SOLO_DISTANCE}({OTHER}, None, {goal_limit}))"
        )
        signals = [
            Signal(OPTIONS, PythonRule(f"((mine - theirs) / (mine + theirs) if {moves} else 0.0)")),
            Signal(MY_OPTIONS, PythonRule(f"(mine / (mine + theirs) if {moves} else 0.0)"), premises=(OPTIONS,)),
            Signal(THEIR_OPTIONS, PythonRule(f"(-theirs / (mine + theirs) if {moves} else 0.0)"), premises=(OPTIONS,)),
            Signal(GOAL_DISTANCE, PythonRule(f"((theirs - mine) / (mine + theirs) if {distances} < {2 * (goal_limit + 1)} else None)")),
        ]
        vocabulary = self._expression_generator.vocabulary(domain, (domain.initial_state,))
        namespace = self._mechanics.variables(domain.initial_state)
        for base, values in vocabulary.values_by_base.items():
            if base.isidentifier() and any(value in players for value in values):
                signals.extend(self._things(base))
                if isinstance(namespace.get(base), Grid):
                    signals.extend(self._grid_things(domain, base))
        for signal in signals:
            logger.info(
                "Derived %s from %s: %s",
                signal.name,
                ", ".join(signal.premises) or "the rules",
                signal.source.source if signal.source is not None else "",
            )
        return tuple(signals)

    def _things(self, base: str) -> tuple[Signal, ...]:
        owned, taking, losing = f"{OWNED} {base}", f"{TAKING} {base}", f"{LOSING} {base}"
        mine, theirs = self._count(HERE, base, ME), self._count(HERE, base, OTHER)
        mine_after, theirs_after = self._count("v1", base, ME), self._count("v1", base, OTHER)
        return (
            Signal(
                owned,
                PythonRule(f"((mine - theirs) / (mine + theirs) if (mine := {mine}) + (theirs := {theirs}) else 0.0)"),
                premises=(OPTIONS,),
            ),
            Signal(
                taking,
                PythonRule(
                    f"(taken / theirs if (taken := {HERE}.best({ME}, lambda v1: ({theirs}) - ({theirs_after}))) > 0 "
                    f"and (theirs := {theirs}) else None)"
                ),
                premises=(owned,),
            ),
            Signal(
                losing,
                PythonRule(
                    f"(-lost / mine if (lost := {HERE}.best({OTHER}, lambda v1: ({mine}) - ({mine_after}))) > 0 "
                    f"and (mine := {mine}) else None)"
                ),
                premises=(owned,),
            ),
            Signal(
                f"{TAKING_MOVES} {base}",
                PythonRule(
                    f"((taken / moves if (moves := {HERE}.mobility({OTHER})) else 0.0) if (taken := {HERE}.best({ME}, "
                    f"lambda v1: {HERE}.mobility({OTHER}) - v1.mobility({OTHER}) if ({theirs_after}) < ({theirs}) "
                    f"else {NOTHING_TAKEN})) > {NOTHING_TAKEN} else None)"
                ),
                premises=(taking, THEIR_OPTIONS),
            ),
            Signal(
                f"{LOSING_MOVES} {base}",
                PythonRule(
                    f"((-lost / moves if (moves := {HERE}.mobility({ME})) else 0.0) if (lost := {HERE}.best({OTHER}, "
                    f"lambda v1: {HERE}.mobility({ME}) - v1.mobility({ME}) if ({mine_after}) < ({mine}) "
                    f"else {NOTHING_TAKEN})) > {NOTHING_TAKEN} else None)"
                ),
                premises=(losing, MY_OPTIONS),
            ),
            Signal(
                f"{FORK} {base}",
                PythonRule(
                    f"(forks / moves if (forks := {HERE}.count({ME}, lambda v1: v1.count({ME}, lambda v2: "
                    f"({theirs_after}) - ({self._count('v2', base, OTHER)}) > 0) >= 2)) > 0 "
                    f"and (moves := {HERE}.mobility({ME})) else None)"
                ),
                premises=(taking,),
            ),
        )

    def _grid_things(self, domain: Domain, base: str) -> tuple[Signal, ...]:
        kinds, worth = self._worth(domain, base)
        logger.info("Worth of a thing of %s by its kind (%s), its average legal moves alone on every cell: %s", base, ", ".join(kinds), worth)
        kind = f"({''.join(f'{HERE}.{other}[at], ' for other in kinds).rstrip(' ')})"
        valued = lambda player: f"sum({worth!r}.get({kind}, 0) for at in {HERE}.{base}.where({player}))"  # noqa: E731
        hanging = (
            f"sum(1 for at in {HERE}.{base}.where({ME}) if {HERE}.changed({OTHER}, {base!r}, at) "
            f"and not {HERE}.with_value({base!r}, at, {OTHER}).changed({ME}, {base!r}, at))"
        )
        return (
            Signal(
                f"{MATERIAL} {base}",
                PythonRule(f"((mine - theirs) / (mine + theirs) if (mine := {valued(ME)}) + (theirs := {valued(OTHER)}) else 0.0)"),
                premises=(f"{OWNED} {base}", f"{TAKING_MOVES} {base}"),
            ),
            Signal(
                f"{HANGING} {base}",
                PythonRule(f"(-hung / owned if (hung := {hanging}) > 0 and (owned := {self._count(HERE, base, ME)}) else None)"),
                premises=(f"{LOSING} {base}",),
            ),
        )

    def _worth(self, domain: Domain, base: str) -> tuple[tuple[str, ...], dict[tuple[Value, ...], float]]:
        """The other grids with the same cells, and each kind's worth: for every player's thing of a kind in the initial
        position, its legal moves alone on each cell, averaged over the players and cells."""
        namespace = self._mechanics.variables(domain.initial_state)
        cells: Grid = namespace[base]  # type: ignore[assignment]
        kinds = tuple(
            sorted(name for name, other in namespace.items() if name != base and isinstance(other, Grid) and other.keys() == cells.keys())
        )
        start = self._mechanics.view(domain, domain.initial_state)
        moves: dict[tuple[Value, ...], list[int]] = {}
        placed: set[tuple[Value, tuple[Value, ...]]] = set()
        for source, owner in sorted(cells.items()):  # type: ignore[type-var]
            if owner not in domain.players.names:
                continue
            kind = tuple(namespace[name][source] for name in kinds)  # type: ignore[index]
            if (owner, kind) in placed:
                continue
            placed.add((owner, kind))
            moves.setdefault(kind, []).extend(start.copied(source, cell).alone(cell).mobility(owner) for cell in cells)  # type: ignore[arg-type]
        return kinds, {kind: round(sum(counts) / len(counts), 3) for kind, counts in moves.items()}

    def _count(self, view: str, base: str, player: str) -> str:
        return f"sum(1 for at in {view}.{base} if {view}.{base}[at] == {player})"
