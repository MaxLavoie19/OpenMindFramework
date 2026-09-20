import logging
import random

from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class PositionGatherer:
    """Positions to reason about, reached the only way any position is reached: by playing the game's own actions from
    where it starts.

    Nothing here knows what a good position is. It walks: from a position, one of the acting players' legal actions,
    drawn at random, and where that leads is the next position. A walk that reaches a position nobody can act in
    starts again from the beginning, so a short game gives many walks and a long one gives few.

    It is what pondering reasons on before a single game has been played, and what it gathers is a sample of the
    positions the game actually reaches — not the start alone, where a game like chess has every piece blocked by its
    own."""

    def gather(self, game: RuleBasedGame, positions: int, seed: int = 0, depth: int | None = None) -> tuple[State, ...]:
        """That many positions, the starting one first, as far as the game offers them. A game that offers no action at
        all gives its starting position alone."""
        rng = random.Random(seed)
        start = game.start()
        gathered: list[State] = [start]
        seen = {start}
        state, steps = start, 0
        while len(gathered) < positions:
            joint = self._played(game, state, rng)
            if joint is None or (depth is not None and steps >= depth):
                if state == start and joint is None:
                    break
                state, steps = start, 0
                continue
            outcomes = game.joint_outcomes(state, joint).outcomes
            if not outcomes:
                state, steps = start, 0
                continue
            state = rng.choices([outcome for outcome, _ in outcomes], weights=[chance for _, chance in outcomes])[0]
            steps += 1
            if state not in seen:
                seen.add(state)
                gathered.append(state)
        logger.info("Gathered %d positions of %s by playing it", len(gathered), game.context)
        return tuple(gathered)

    def _played(self, game: RuleBasedGame, state: State, rng: random.Random) -> JointAction | None:
        """One action drawn for every player who can act here, taken together: all players play at once, so a walk
        that moved one of them at a time would ask a game what half a turn leads to. None where nobody can act."""
        players = game.players().names
        legal = game.joint_actions(state)
        if not legal:
            return None
        return JointAction(tuple((players[index], rng.choice(actions)) for index, actions in legal))
