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
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value
from openmind.world.service.state_reader import StateReader


class ConsequenceLibrary:
    """What generated rules read besides a state's variables, for any game, worked out with the RBS's own legal moves and
    predictor:

    - `me` and `other`: the player to act, or the player a position is valued for, and the next player in the game's
      order;
    - `win_chance(action)`: the probability that the action ends the game in a win for the player taking it;
    - `wins(player, action=None)`: the summed win chances of the actions `player` could take if it were their turn, now
      or, expected over its outcomes, after `action`;
    - `near(action, *offset)`: the value, now, of the variable at that index offset from the indexed variable the action
      sets, or `OUTSIDE`;
    - `here`: the position as the mechanics' view, which looks ahead with the game's actions (see
      `inference/README.md`).

    A win is an outcome with no legal action left in which the player's payoff is higher than every other player's."""

    def __init__(
        self,
        state_reader: StateReader,
        variable_name_mapper: VariableNameMapper,
        mechanics: Mechanics,
    ) -> None:
        self._state_reader = state_reader
        self._variable_name_mapper = variable_name_mapper
        self._mechanics = mechanics
        self._games: dict[int, RuleBasedSystem] = {}
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

    def names(self, rbs: RuleBasedSystem, state: State, player: str | None = None) -> dict[str, object]:
        """The names for a state, `me` being the player to act or, when given, that player; the same mapping every time
        the state and player come back."""
        key = ("names", self._pin(rbs), state, player)
        names = self._cache.get(key)
        if names is None:
            me = dict(state.variables).get(rbs.players().to_act) if player is None else player
            players = rbs.players().names
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

    def win_chance(self, rbs: RuleBasedSystem, state: State, action: Action) -> float:
        key = ("win_chance", self._pin(rbs), state, action)
        chance = self._cache.get(key)
        if chance is None:
            mover = self._state_reader.player_to_act(state, rbs.players())
            chance = math.fsum(
                probability
                for outcome, probability in rbs.outcomes(state, action).outcomes
                if self.is_win(rbs, outcome, mover)
            )
            self._remember(key, chance)
        return chance  # type: ignore[return-value]

    def wins(self, rbs: RuleBasedSystem, state: State, player: str, action: Action | None = None) -> float:
        if action is not None:
            return math.fsum(
                probability * self.wins(rbs, outcome, player)
                for outcome, probability in rbs.outcomes(state, action).outcomes
            )
        key = ("wins", self._pin(rbs), state, player)
        count = self._cache.get(key)
        if count is None:
            turned = self.with_turn(rbs, state, player)
            count = math.fsum(
                self.win_chance(rbs, turned, candidate) for candidate in rbs.actions(turned)
            )
            self._remember(key, count)
        return count  # type: ignore[return-value]

    def near(self, rbs: RuleBasedSystem, state: State, action: Action, offset: tuple[int, ...]) -> Value:
        anchor = self.anchor(rbs, state, action)
        if anchor is None or len(anchor[1]) != len(offset):
            return OUTSIDE
        base, indices = anchor
        name = self._variable_name_mapper.to_name(base, tuple(index + step for index, step in zip(indices, offset)))
        return self._variables(rbs, state).get(name, OUTSIDE)

    def anchor(self, rbs: RuleBasedSystem, state: State, action: Action) -> tuple[str, tuple[int, ...]] | None:
        """The base and whole-number indices of the first indexed variable, other than the players' variables, whose
        value the action's first outcome changes; None when there is none."""
        key = ("anchor", self._pin(rbs), state, action)
        if key not in self._cache:
            anchor = None
            players = {rbs.players().to_act, *rbs.players().payoffs}
            outcome = rbs.outcomes(state, action).outcomes[0][0]
            previous = dict(state.variables)
            for name, after in outcome.variables:
                if name in players or (name in previous and after == previous[name]):
                    continue
                base, texts = self._variable_name_mapper.from_name(name)
                if texts and all(text.lstrip("-").isdecimal() for text in texts):
                    anchor = (base, tuple(int(text) for text in texts))
                    break
            self._remember(key, anchor)
        return self._cache[key]  # type: ignore[return-value]

    def is_win(self, rbs: RuleBasedSystem, state: State, player: int) -> bool:
        """Whether no legal action is left and the player's payoff is higher than every other player's. The payoffs are
        read first: a state with a payoff that isn't a number, or where the player's isn't the highest, is no win without
        solving for its legal actions."""
        try:
            payoffs = self._state_reader.payoffs(state, rbs.players())
        except ValueError:
            return False
        if not all(payoffs[player] > payoff for index, payoff in enumerate(payoffs) if index != player):
            return False
        return not rbs.actions(state)

    def with_turn(self, rbs: RuleBasedSystem, state: State, player: str) -> State:
        """The state with the player to act replaced."""
        return State(
            tuple((name, player if name == rbs.players().to_act else value) for name, value in state.variables)
        )

    def _variables(self, rbs: RuleBasedSystem, state: State) -> dict[str, Value]:
        key = ("variables", self._pin(rbs), state)
        variables = self._cache.get(key)
        if variables is None:
            variables = dict(state.variables)
            self._remember(key, variables)
        return variables  # type: ignore[return-value]

    def _pin(self, rbs: RuleBasedSystem) -> int:
        """An identity for the game that stays valid: the RBS is kept alive as long as the library."""
        key = id(rbs)
        if key not in self._games:
            self._games[key] = rbs
        return key

    def _remember(self, key: tuple[object, ...], value: object) -> None:
        self._memory_guard.remembered()
        self._cache[key] = value
