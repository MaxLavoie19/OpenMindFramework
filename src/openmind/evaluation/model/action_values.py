from openmind.world.model.action import Action

#: Every legal action of a position with its value for the player to act, in the solver's order.
type ActionValues = tuple[tuple[Action, float], ...]
