import logging

from openmind.agent.service.game_memory import GameMemory
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.model.studied import Studied
from openmind.training.service.game_replayer import GameReplayer
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class GameStudy:
    """The positions of games already played, read back so they can be learned from afterwards.

    **Playing is timely and studying is not, and that is the whole of why these are two things.** A move has to
    come back before the clock runs out, so nothing may be learned inside one. What can be done inside a move is
    to write down what happened, which is what `GameMemory` has been doing all along — every finished game, its
    actions and its seeds, kept as it ended. Reading it back costs whatever it costs, because by then nobody is
    waiting.

    **What comes back is the same evidence a scout brings.** A studied position can be put to the game exactly
    as a walked-to one can: what does it allow here. So learning from games played is not a second kind of
    learning beside learning from positions found — it is the same learning with the positions arriving from
    somewhere else. That holds for games the agent played against itself and for games somebody else played,
    which is the only reason watching another player is not a harder problem than walking the game oneself.

    Nothing here knows which games are worth studying. It gives them back in the order they ended, and a caller
    that wants the recent ones, or the decisive ones, or a sample, says so."""

    def __init__(self, game_replayer: GameReplayer | None = None) -> None:
        self._replayer = GameReplayer() if game_replayer is None else game_replayer

    def studied(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        kind: str | None = None,
        most: int | None = None,
    ) -> tuple[Studied, ...]:
        """Every position of every game remembered there, in the order they were played, each carrying what
        the game it came from came to.

        **The position without its game has lost what makes it worth learning from.** Who played it and how
        it ended are facts about the position as much as about the game, and both were already written down
        when the game ended — a position handed back bare threw them away on the way out.

        `kind` keeps to games of one kind, and `most` to the last so many games — the last, because a game
        played recently was played with whatever is believed now.

        A game that cannot be replayed is left out rather than stopping the study: a game where several
        players acted at once does not say who did what, and one remembered without its outcome seed cannot
        have its chances drawn again."""
        summaries = GameMemory(knowledge_base).games(kind)
        wanted = summaries if most is None else summaries[-most:]
        found: list[Studied] = []
        replayed = 0
        for summary in wanted:
            try:
                states = self._replayer.positions(game, summary)
            except ValueError:
                logger.debug("Left %s out of the study: it cannot be played again", summary.label)
                continue
            if len(states) < 2:
                continue
            replayed += 1
            found.extend(
                Studied(
                    state,
                    ply,
                    summary.label,
                    tuple(summary.players),
                    tuple(one.name for one in summary.models),
                    tuple(summary.payoffs),
                    summary.ending or "",
                )
                for ply, state in enumerate(states)
            )
        logger.info(
            "Studied %d of %d games remembered of %s: %d positions to learn from, %d of games that ended %s",
            replayed,
            len(wanted),
            game.context,
            len(found),
            sum(1 for one in found if one.ending),
            ", ".join(sorted({one.ending for one in found if one.ending})) or "in no stated way",
        )
        return tuple(found)

    def positions(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        kind: str | None = None,
        most: int | None = None,
    ) -> tuple[State, ...]:
        """The same, as bare positions, for a caller that wants only the boards."""
        return tuple(one.state for one in self.studied(knowledge_base, game, kind, most))
