from openmind.structure.model.map import Map
from openmind.structure.model.value import Value
from openmind.world.model.players import Players
from openmind.world.model.state import State


class StateReader:
    """Reads a state: a scalar's value by name, and the payoffs."""

    def value(self, state: State, name: str) -> Value:
        return state.value(name)

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
