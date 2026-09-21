import logging
import time
from collections.abc import Sequence

from openmind.inference.model.sides import Sides
from openmind.inference.model.worth import Worth
from openmind.structure.model.grid import Grid
from openmind.structure.model.value import Value
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class WorthReasoner:
    """Works out what the things on a board are worth, before a single game is played.

    A heuristic has to come from somewhere, and the two ways OMF had both need something it does not have at the
    start: fitting needs positions whose value is already known, and telling proposals apart by playing them needs
    games. What it does have is the rules, and the rules say what a thing is for.

    So a thing is valued by what having it lets its owner do. Take it off the board, ask the game what its owner
    can do now, and the difference is what it afforded. A queen afforded a great deal, a pawn little, and nobody
    said so — the movement rules did, by being asked.

    That more affordance is better is not assumed either. Some of the positions gathered are over, and the game
    says what it paid in them. Where the worst-paid player in those positions is the one who could do nothing, the
    game itself has said that being unable to act is the worst thing there is, and valuing what lets you act
    follows from it. Where the positions say otherwise, the worths are still counted and reported as resting on
    nothing, because a game where being able to do less is better is a game this reasoning does not fit."""

    def reason(
        self,
        game: object,
        positions: Sequence[State],
        sides: Sides | None = None,
        most: int | None = None,
    ) -> Worth:
        """What each thing on the board is worth to whoever owns it.

        `sides` says which values belong to which player, so a thing is only ever valued to its owner; without it
        every thing is valued to whoever is acting, which is right for a game where the acting player owns
        everything they can move. `most` is how many positions to reason over, since asking the rules what is legal
        is the expensive part."""
        started = time.monotonic()
        looked = list(positions if most is None else positions[:most])
        settled, ended = self._settled(game, looked)
        taken: dict[tuple[str, Value], list[float]] = {}
        for state in looked:
            if game.ended(state) is not None:  # type: ignore[attr-defined]
                continue
            acting = self._acting(game, state)
            if acting is None:
                continue
            standing = len(self._actions(game, state, acting))
            if not standing:
                continue
            for model, grid in self._grids(state).items():
                for at in grid.coordinates():
                    held = grid.at(at)
                    if held is None or not self._owned(sides, model, held, acting):
                        continue
                    without = self._actions(game, state.with_model(model, grid.removed(at)), acting)
                    taken.setdefault((model, held), []).append(standing - len(without))
        holdings = tuple(
            sorted(
                ((model, value, sum(lost) / len(lost)) for (model, value), lost in taken.items() if lost),
                key=lambda one: -one[2],
            )
        )
        worth = Worth(holdings, ended, settled, time.monotonic() - started)
        logger.info(
            "Reasoned out what %d kinds of thing are worth over %d positions in %.0f seconds, %s",
            len(holdings),
            len(looked),
            worth.seconds,
            f"borne out by {ended} positions that were over" if settled else "borne out by nothing",
        )
        return worth

    def _settled(self, game: object, positions: Sequence[State]) -> tuple[bool, int]:
        """Whether the positions that were over say that being unable to act is being worst off.

        This is the whole ground of the reasoning, and it is read rather than assumed: in every position the game
        has finished, the player it paid least is asked what they could have done. Where that is nothing, every
        time, the game has said it."""
        ended = borne = 0
        for state in positions:
            payoffs = game.ended(state)  # type: ignore[attr-defined]
            if payoffs is None:
                continue
            ended += 1
            worst = min(payoffs.values()) if hasattr(payoffs, "values") else None
            if worst is None:
                continue
            losing = [player for player in payoffs.keys() if payoffs[player] == worst]  # type: ignore[attr-defined]
            if all(not self._actions(game, state, player) for player in losing):
                borne += 1
        return (ended > 0 and borne == ended), ended

    def _actions(self, game: object, state: State, player: Value) -> tuple:
        try:
            return tuple(game.actions(state, player=player))  # type: ignore[attr-defined]
        except TypeError:
            return tuple(game.actions(state))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a position the rules cannot answer about affords nothing
            return ()

    def _acting(self, game: object, state: State) -> Value | None:
        acting = game.joint_actions(state)  # type: ignore[attr-defined]
        if not acting:
            return None
        return game.players().names[acting[0][0]]  # type: ignore[attr-defined]

    def _owned(self, sides: Sides | None, model: str, value: Value, acting: Value) -> bool:
        """Whether that thing is the acting player's, where anything says whose things are whose."""
        if sides is None:
            return True
        whose = sides.whose(model, value)
        return whose is None or whose == acting

    def _grids(self, state: State) -> dict[str, Grid]:
        return {name: model for name, model in state.models if isinstance(model, Grid)}
