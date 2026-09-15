from openmind.csp.model.problem import Problem
from openmind.csp.service.solver import Solver
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class JointSolver:
    """The legal actions of every player to act in a state, one solve per player, the problem's rules reading that player
    as `player`: what players acting at once choose from."""

    def __init__(self, solver: Solver, state_reader: StateReader) -> None:
        self._solver = solver
        self._state_reader = state_reader

    def legal(self, problem: Problem, state: State, players: Players) -> tuple[tuple[int, tuple[Action, ...]], ...]:
        """Each player to act, by index in players.names, with its legal actions; empty when the game is over, no player
        to act having any. Players to act without a legal action while others have one raise ValueError."""
        legal = tuple(
            (index, self._solver.solve(problem, state, player=players.names[index]))
            for index in self._state_reader.players_to_act(state, players)
        )
        stuck = [players.names[index] for index, actions in legal if not actions]
        if not stuck:
            return legal
        if len(stuck) == len(legal):
            return ()
        raise ValueError(f"{', '.join(stuck)} can't act while other players to act can: every player to act needs an action")
