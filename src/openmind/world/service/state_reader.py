from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value


class StateReader:
    """Reads a state's variables: any value by name, the players to act, and the payoffs. The players to act are named
    by the to_act variable, or, in a domain where players act at once, flagged by one variable per player under the
    to_act base: `turn(A)` and `turn(B)` true when both act."""

    def value(self, state: State, name: str) -> Value:
        for key, value in state.variables:
            if key == name:
                return value
        raise KeyError(f"Unknown state variable: {name!r}")

    def acts_at_once(self, state: State, players: Players) -> bool:
        """Whether the players to act are flagged, one variable per player under the to_act base, instead of named."""
        names = {name for name, _ in state.variables}
        return players.to_act not in names and any(self._flag(players, player) in names for player in players.names)

    def players_to_act(self, state: State, players: Players) -> tuple[int, ...]:
        """The indices in players.names of the players to act: the one the to_act variable names, or every player whose
        flag is true, none when the game is over."""
        if not self.acts_at_once(state, players):
            return (self.player_to_act(state, players),)
        variables = dict(state.variables)
        return tuple(index for index, player in enumerate(players.names) if variables.get(self._flag(players, player)) is True)

    def player_to_act(self, state: State, players: Players) -> int:
        """The index in players.names of the one player to act; players acting at once raise ValueError."""
        for key, name in state.variables:
            if key == players.to_act:
                if not isinstance(name, str) or name not in players.names:
                    raise ValueError(f"{players.to_act} is {name!r}, not one of {players.names}")
                return players.names.index(name)
        if not self.acts_at_once(state, players):
            raise KeyError(f"Unknown state variable: {players.to_act!r}")
        acting = self.players_to_act(state, players)
        if len(acting) != 1:
            raise ValueError(
                f"{', '.join(players.names[index] for index in acting) or 'No player'} act at once, not one player to act"
            )
        return acting[0]

    def payoffs(self, state: State, players: Players) -> tuple[float, ...]:
        """Each player's payoff, in the order of players.names."""
        payoffs: list[float] = []
        for name in players.payoffs:
            value = self.value(state, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} is {value!r}, not a number")
            payoffs.append(float(value))
        return tuple(payoffs)

    def _flag(self, players: Players, player: str) -> str:
        return f"{players.to_act}({player})"
