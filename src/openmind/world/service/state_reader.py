from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.structure.model.value import Value
from openmind.world.model.players import Players
from openmind.world.model.state import State


class StateReader:
    """Reads a state: a scalar's value by name, the players to act, and the payoffs. The players to act are named by the
    to_act Scalar, or, where players act at once, flagged in the to_act Map: `turn = {A: True, B: True}` when both act."""

    def value(self, state: State, name: str) -> Value:
        return state.value(name)

    def acts_at_once(self, state: State, players: Players) -> bool:
        """Whether the players to act are flagged in a Map rather than named by a Scalar."""
        return state.has(players.to_act) and isinstance(state.model(players.to_act), Map)

    def players_to_act(self, state: State, players: Players) -> tuple[int, ...]:
        """The indices in players.names of the players to act: the one the to_act Scalar names, or every player the
        to_act Map flags true, none when the game is over."""
        if not self.acts_at_once(state, players):
            return (self.player_to_act(state, players),)
        flags = state.model(players.to_act)
        assert isinstance(flags, Map)
        return tuple(index for index, player in enumerate(players.names) if flags.get(player) is True)

    def player_to_act(self, state: State, players: Players) -> int:
        """The index in players.names of the one player to act; players acting at once raise ValueError unless exactly
        one is flagged."""
        model = state.model(players.to_act)
        if isinstance(model, Scalar):
            name = model.value
            if not isinstance(name, str) or name not in players.names:
                raise ValueError(f"{players.to_act} is {name!r}, not one of {players.names}")
            return players.names.index(name)
        if not isinstance(model, Map):
            raise TypeError(f"{players.to_act} is a {type(model).__name__}; it must be a Scalar or a Map of flags")
        acting = self.players_to_act(state, players)
        if len(acting) != 1:
            raise ValueError(
                f"{', '.join(players.names[index] for index in acting) or 'No player'} act at once, not one player to act"
            )
        return acting[0]

    def payoffs(self, state: State, players: Players) -> tuple[float, ...]:
        """Each player's payoff, in the order of players.names."""
        held = state.model(players.payoff)
        if not isinstance(held, Map):
            raise TypeError(f"{players.payoff} is a {type(held).__name__}, not a Map")
        payoffs: list[float] = []
        for player in players.names:
            value = held.get(player)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{players.payoff}[{player!r}] is {value!r}, not a number")
            payoffs.append(float(value))
        return tuple(payoffs)
