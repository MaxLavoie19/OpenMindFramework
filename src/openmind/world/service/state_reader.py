from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value


class StateReader:
    """Reads a state's variables: any value by name, the player to act, and the payoffs."""

    def value(self, state: State, name: str) -> Value:
        for key, value in state.variables:
            if key == name:
                return value
        raise KeyError(f"Unknown state variable: {name!r}")

    def player_to_act(self, state: State, players: Players) -> int:
        """The index in players.names of the player named by the to_act variable."""
        name = self.value(state, players.to_act)
        if not isinstance(name, str) or name not in players.names:
            raise ValueError(f"{players.to_act} is {name!r}, not one of {players.names}")
        return players.names.index(name)

    def payoffs(self, state: State, players: Players) -> tuple[float, ...]:
        """Each player's payoff, in the order of players.names."""
        payoffs: list[float] = []
        for name in players.payoffs:
            value = self.value(state, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} is {value!r}, not a number")
            payoffs.append(float(value))
        return tuple(payoffs)
