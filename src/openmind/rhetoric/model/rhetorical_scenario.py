from dataclasses import dataclass

from openmind.rhetoric.model.rhetorical_goal import RhetoricalGoal
from openmind.rhetoric.model.speaker import Speaker


@dataclass(frozen=True, slots=True)
class RhetoricalScenario:
    """A situation to plan in: the speaker as they know themselves and perceive their audience, the goal they pursue, and
    how many strategic moves they make before the exchange is scored. Every position the speaker holds, shows or
    perceives answers the same questions."""

    name: str
    speaker: Speaker
    goal: RhetoricalGoal
    moves: int
