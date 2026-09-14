from collections.abc import Callable

from openmind.agent.model.policy import Policy

#: Gives the policy that plays a game, from the game's seed; to run in worker processes, it must pickle.
type PolicyFactory = Callable[[int], Policy]
