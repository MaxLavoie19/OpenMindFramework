import math

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.predictor.service.predictor import Predictor
from openmind.rbs.constant.consequence_constant import (
    CACHE_SIZE,
    ME,
    NEAR,
    OTHER,
    OUTSIDE,
    OUTSIDE_NAME,
    SOLO_DISTANCE,
    WIN_CHANCE,
    WINS,
)
from openmind.inference.constant.inference_constant import HERE
from openmind.inference.service.mechanics import Mechanics
from openmind.rbs.constant.generation_constant import DEFAULT_SOLO_LIMIT
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value
from openmind.world.service.state_reader import StateReader


class ConsequenceLibrary:
    """What generated rules read besides a state's variables, for any domain, worked out with the domain's own solver and
    predictor:

    - `me` and `other`: the player to act, or the player a position is valued for, and the next player in the domain's
      order;
    - `win_chance(action)`: the probability that the action ends the game in a win for the player taking it;
    - `wins(player, action=None)`: the summed win chances of the actions `player` could take if it were their turn, now
      or, expected over its outcomes, after `action`;
    - `solo_distance(player, action=None, limit=2)`: the fewest of `player`'s own actions after which a win is possible,
      if nobody else moved, now or after `action`; `limit + 1` when none is found;
    - `near(action, *offset)`: the value, now, of the variable at that index offset from the indexed variable the action
      sets, or `OUTSIDE`;
    - `here`: the position as the mechanics' view, which looks ahead with the domain's actions (see
      `inference/README.md`).

    A win is an outcome with no legal action left in which the player's payoff is higher than every other player's."""

    def __init__(
        self,
        solver: Solver,
        predictor: Predictor,
        state_reader: StateReader,
        variable_name_mapper: VariableNameMapper,
        mechanics: Mechanics,
    ) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._variable_name_mapper = variable_name_mapper
        self._mechanics = mechanics
        self._domains: dict[int, Domain] = {}
        self._cache: dict[tuple[object, ...], object] = {}

    def __getstate__(self) -> dict[str, object]:
        """The lookups stay behind when the library is copied to another process: they are keyed by this process's ids
        and hold functions."""
        return {name: value for name, value in self.__dict__.items() if name not in ("_domains", "_cache")}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._domains = {}
        self._cache = {}

    def limit_memory(self, memory_bytes: int) -> None:
        """How many bytes a process holds before the mechanics clear their views; copies sent to workers carry it."""
        self._mechanics.limit_memory(memory_bytes)

    def clear_memory(self) -> None:
        """Forgets the lookups and the mechanics' views this process kept."""
        self._cache.clear()
        self._mechanics.clear()

    def names(self, domain: Domain, state: State, player: str | None = None) -> dict[str, object]:
        """The names for a state, `me` being the player to act or, when given, that player; the same mapping every time
        the state and player come back."""
        key = ("names", self._pin(domain), state, player)
        names = self._cache.get(key)
        if names is None:
            me = dict(state.variables).get(domain.players.to_act) if player is None else player
            players = domain.players.names
            other = players[(players.index(me) + 1) % len(players)] if me in players else None
            names = {
                ME: me,
                OTHER: other,
                OUTSIDE_NAME: OUTSIDE,
                WIN_CHANCE: lambda action: self.win_chance(domain, state, action),
                WINS: lambda player, action=None: self.wins(domain, state, player, action),
                SOLO_DISTANCE: lambda player, action=None, limit=DEFAULT_SOLO_LIMIT: self.solo_distance(
                    domain, state, player, action, limit
                ),
                NEAR: lambda action, *offset: self.near(domain, state, action, offset),
                HERE: self._mechanics.view(domain, state),
            }
            self._remember(key, names)
        return names  # type: ignore[return-value]

    def win_chance(self, domain: Domain, state: State, action: Action) -> float:
        key = ("win_chance", self._pin(domain), state, action)
        chance = self._cache.get(key)
        if chance is None:
            mover = self._state_reader.player_to_act(state, domain.players)
            chance = math.fsum(
                probability
                for outcome, probability in self._predictor.predict(domain.transitions, state, action).outcomes
                if self.is_win(domain, outcome, mover)
            )
            self._remember(key, chance)
        return chance  # type: ignore[return-value]

    def wins(self, domain: Domain, state: State, player: str, action: Action | None = None) -> float:
        if action is not None:
            return math.fsum(
                probability * self.wins(domain, outcome, player)
                for outcome, probability in self._predictor.predict(domain.transitions, state, action).outcomes
            )
        key = ("wins", self._pin(domain), state, player)
        count = self._cache.get(key)
        if count is None:
            turned = self.with_turn(domain, state, player)
            count = math.fsum(
                self.win_chance(domain, turned, candidate) for candidate in self._solver.solve(domain.problem, turned)
            )
            self._remember(key, count)
        return count  # type: ignore[return-value]

    def solo_distance(
        self,
        domain: Domain,
        state: State,
        player: str,
        action: Action | None = None,
        limit: int = DEFAULT_SOLO_LIMIT,
    ) -> float:
        if action is not None:
            return math.fsum(
                probability * self.solo_distance(domain, outcome, player, None, limit)
                for outcome, probability in self._predictor.predict(domain.transitions, state, action).outcomes
            )
        turned = self.with_turn(domain, state, player)
        for depth in range(1, limit + 1):
            if self._reaches(domain, turned, player, depth):
                return depth
        return limit + 1

    def near(self, domain: Domain, state: State, action: Action, offset: tuple[int, ...]) -> Value:
        anchor = self.anchor(domain, state, action)
        if anchor is None or len(anchor[1]) != len(offset):
            return OUTSIDE
        base, indices = anchor
        name = self._variable_name_mapper.to_name(base, tuple(index + step for index, step in zip(indices, offset)))
        return self._variables(domain, state).get(name, OUTSIDE)

    def anchor(self, domain: Domain, state: State, action: Action) -> tuple[str, tuple[int, ...]] | None:
        """The base and whole-number indices of the first indexed variable, other than the players' variables, whose
        value the action's first outcome changes; None when there is none."""
        key = ("anchor", self._pin(domain), state, action)
        if key not in self._cache:
            anchor = None
            players = {domain.players.to_act, *domain.players.payoffs}
            outcome = self._predictor.predict(domain.transitions, state, action).outcomes[0][0]
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

    def is_win(self, domain: Domain, state: State, player: int) -> bool:
        """Whether no legal action is left and the player's payoff is higher than every other player's. The payoffs are
        read first: a state with a payoff that isn't a number, or where the player's isn't the highest, is no win without
        solving for its legal actions."""
        try:
            payoffs = self._state_reader.payoffs(state, domain.players)
        except ValueError:
            return False
        if not all(payoffs[player] > payoff for index, payoff in enumerate(payoffs) if index != player):
            return False
        return not self._solver.solve(domain.problem, state)

    def with_turn(self, domain: Domain, state: State, player: str) -> State:
        """The state with the player to act replaced."""
        return State(
            tuple((name, player if name == domain.players.to_act else value) for name, value in state.variables)
        )

    def _reaches(self, domain: Domain, state: State, player: str, depth: int) -> bool:
        key = ("reaches", self._pin(domain), state, player, depth)
        if key not in self._cache:
            reaches = False
            for candidate in self._solver.solve(domain.problem, state):
                if depth == 1:
                    reaches = self.win_chance(domain, state, candidate) > 0
                else:
                    reaches = any(
                        self._reaches(domain, self.with_turn(domain, outcome, player), player, depth - 1)
                        for outcome, probability in self._predictor.predict(domain.transitions, state, candidate).outcomes
                        if probability > 0
                    )
                if reaches:
                    break
            self._remember(key, reaches)
        return self._cache[key]  # type: ignore[return-value]

    def _variables(self, domain: Domain, state: State) -> dict[str, Value]:
        key = ("variables", self._pin(domain), state)
        variables = self._cache.get(key)
        if variables is None:
            variables = dict(state.variables)
            self._remember(key, variables)
        return variables  # type: ignore[return-value]

    def _pin(self, domain: Domain) -> int:
        """An identity for the domain that stays valid: the domain is kept alive as long as the library."""
        self._domains.setdefault(id(domain), domain)
        return id(domain)

    def _remember(self, key: tuple[object, ...], value: object) -> None:
        if len(self._cache) >= CACHE_SIZE:
            self._cache.clear()
        self._cache[key] = value
