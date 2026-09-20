from dataclasses import dataclass

from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One guess at what an agent can't see, with how likely it is.

    Either the hidden part filled in — the cards that player holds — or the policy that agent is following, where the
    possibilities can't be counted: under the fog of war an opponent can't be enumerated, but they are going economy,
    nuclear or swarm, and the odds shift as the game shows which."""

    likelihood: float
    state: State | None = None
    policy: str | None = None
