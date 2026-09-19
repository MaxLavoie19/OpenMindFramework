import math

from openmind.inference.constant.inference_constant import HERE
from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.rbs.constant.consequence_constant import (
    ME,
    NEAR,
    OTHER,
    OUTSIDE,
    OUTSIDE_NAME,
    WIN_CHANCE,
    WINS,
)
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.structure.model.grid import Grid
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.structure.model.value import Value
from openmind.world.service.state_reader import StateReader


class ConsequenceLibrary:
    """What generated rules read besides a state's variables, for any game, worked out with the RBS's own legal moves and
    predictor:

    - `me` and `other`: the one player acting, or the player a position is valued for, and the next player in the
      game's order;
    - `win_chance(action)`: the probability that the action ends the game in a win for the player taking it;
    - `wins(player, action=None)`: the summed win chances of the actions `player` can take, now or, expected over its
      outcomes, after `action`; none where the game gives the player no action;
    - `near(action, *offset)`: the value, now, of the variable at that index offset from the indexed variable the action
      sets, or `OUTSIDE`;
    - `here`: the position as the mechanics' view, which looks ahead with the game's actions (see
      `inference/README.md`).

    A win is an outcome with no legal action left in which the player's payoff is higher than every other player's."""

    def __init__(
        self,
        state_reader: StateReader,
        mechanics: Mechanics,
    ) -> None:
        self._state_reader = state_reader
        self._mechanics = mechanics
        self._games: dict[int, RuleBasedGame] = {}
        self._cache: dict[tuple[object, ...], object] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        """The lookups stay behind when the library is copied to another process: they are keyed by this process's ids
        and hold functions."""
        return {
            name: value for name, value in self.__dict__.items() if name not in ("_games", "_cache", "_memory_guard")
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._games = {}
        self._cache = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def memory_entries(self) -> int:
        return len(self._cache)

    def evict_memory(self, entries: int) -> None:
        """Keeps the newest lookups only; the mechanics keep their own to the count the guard gives them."""
        evict_oldest(self._cache, entries)

    def limit_memory(self, memory_bytes: int) -> None:
        """How many bytes a process holds before the mechanics clear their views; copies sent to workers carry it."""
        self._mechanics.limit_memory(memory_bytes)

    def clear_memory(self) -> None:
        """Forgets the lookups and the mechanics' views this process kept."""
        self._cache.clear()
        self._mechanics.clear()

    def names(self, rbs: RuleBasedGame, state: State, player: str | None = None) -> dict[str, object]:
        """The names for a state, `me` being the one player acting there or, when given, that player; the same mapping
        every time the state and player come back."""
        key = ("names", self._pin(rbs), state, player)
        names = self._cache.get(key)
        if names is None:
            players = rbs.players().names
            acting = rbs.acting(state) if player is None else ()
            me = player if player is not None else (players[acting[0]] if len(acting) == 1 else None)
            other = players[(players.index(me) + 1) % len(players)] if me in players else None
            names = {
                ME: me,
                OTHER: other,
                OUTSIDE_NAME: OUTSIDE,
                WIN_CHANCE: lambda action: self.win_chance(rbs, state, action),
                WINS: lambda player, action=None: self.wins(rbs, state, player, action),
                NEAR: lambda action, *offset: self.near(rbs, state, action, offset),
                HERE: self._mechanics.view(rbs, state),
            }
            self._remember(key, names)
        return names  # type: ignore[return-value]

    def win_chance(self, rbs: RuleBasedGame, state: State, action: Action) -> float:
        key = ("win_chance", self._pin(rbs), state, action)
        chance = self._cache.get(key)
        if chance is None:
            mover = rbs.players().names.index(rbs.acting_player(state))
            chance = math.fsum(
                probability
                for outcome, probability in rbs.outcomes(state, action).outcomes
                if self.is_win(rbs, outcome, mover)
            )
            self._remember(key, chance)
        return chance  # type: ignore[return-value]

    def wins(self, rbs: RuleBasedGame, state: State, player: str, action: Action | None = None) -> float:
        if action is not None:
            return math.fsum(
                probability * self.wins(rbs, outcome, player)
                for outcome, probability in rbs.outcomes(state, action).outcomes
            )
        key = ("wins", self._pin(rbs), state, player)
        count = self._cache.get(key)
        if count is None:
            count = math.fsum(
                self.win_chance(rbs, state, candidate) for candidate in rbs.actions(state, player=player)
            )
            self._remember(key, count)
        return count  # type: ignore[return-value]

    def near(self, rbs: RuleBasedGame, state: State, action: Action, offset: tuple[int, ...]) -> Value:
        anchor = self.anchor(rbs, state, action)
        if anchor is None or len(anchor[1]) != len(offset):
            return OUTSIDE
        name, coordinates = anchor
        grid = state.model(name)
        where = tuple(index + step for index, step in zip(coordinates, offset, strict=True))
        return grid.at(where) if isinstance(grid, Grid) and grid.inside(where) else OUTSIDE

    def anchor(self, rbs: RuleBasedGame, state: State, action: Action) -> tuple[str, tuple[int, ...]] | None:
        """The grid and the coordinates of the first cell the action's first outcome changes, grids read in the order
        of their names and cells row-major; None when no grid changes."""
        key = ("anchor", self._pin(rbs), state, action)
        if key not in self._cache:
            anchor = None
            outcome = rbs.outcomes(state, action).outcomes[0][0]
            for name, after in outcome.models:
                before = state.model(name) if state.has(name) else None
                if not isinstance(after, Grid) or not isinstance(before, Grid) or after == before:
                    continue
                changed = next(
                    (where for where, value in after.items() if before.inside(where) and before.at(where) != value), None
                )
                if changed is not None:
                    anchor = (name, changed)
                    break
            self._remember(key, anchor)
        return self._cache[key]  # type: ignore[return-value]

    def is_win(self, rbs: RuleBasedGame, state: State, player: int) -> bool:
        """Whether no player has a legal action left and the player's payoff is higher than every other player's. The
        payoffs are read first: a state with a payoff that isn't a number, or where the player's isn't the highest, is no
        win without solving for its legal actions."""
        try:
            payoffs = self._state_reader.payoffs(state, rbs.players())
        except ValueError:
            return False
        if not all(payoffs[player] > payoff for index, payoff in enumerate(payoffs) if index != player):
            return False
        return not rbs.acting(state)

    def _pin(self, rbs: RuleBasedGame) -> int:
        """An identity for the game that stays valid: the RBS is kept alive as long as the library."""
        key = id(rbs)
        if key not in self._games:
            self._games[key] = rbs
        return key

    def _remember(self, key: tuple[object, ...], value: object) -> None:
        self._memory_guard.remembered()
        self._cache[key] = value
