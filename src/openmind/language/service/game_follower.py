import logging
from collections.abc import Callable, Sequence

from openmind.language.model.following import Following
from openmind.language.model.grammar import Grammar
from openmind.language.model.happening import Happening
from openmind.language.service.decoder import Decoder
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: Why a game stopped being followable, in the words a reader of the log needs.
REFUSED = "the rules refuse a move that was played"
SEVERAL = "the notation does not tell those moves apart"
NOWHERE = "the move could not be made"


class GameFollower:
    """The positions of a game somebody else played, worked out from the notation they wrote it in.

    **This is how a game nobody here played becomes something to learn from.** A game the agent played is
    remembered with its actions and can be replayed outright. A game somebody else played arrives as writing —
    `1. e4 e5 2. Nf3` — and the actions have to be recovered from it. Everything after that is the same: a
    position, the game asked what it allows there, and the difference between the two.

    The recovery is the decoder's, and it was built for a different purpose. A run learns to read its own
    game's notation in order to *check* what it learned — what it would write against what was written. The
    same reading, pointed at writing from outside, follows a game it never saw.

    **Nothing here knows a game, a predictor or a drawer.** It is handed a way to ask what could happen at a
    position and a way to take one of those, and it walks. That keeps it usable where the actions are recovered
    some other way, and it keeps the question of what a move does where that question already lives.

    **Where it stops is the finding.** A notation narrowing to nothing means the rules refuse a move somebody
    really played, which is a rule too tight — the one mistake that never shows up in play, because the agent
    simply never makes that move. A notation narrowing to several means the reading cannot tell those moves
    apart yet. Both are worth knowing and they want opposite work, so it says which."""

    def __init__(self, decoder: Decoder | None = None) -> None:
        self._decoder = Decoder() if decoder is None else decoder

    def follow(
        self,
        start: State,
        notation: Sequence[str],
        happenings: Callable[[State], Sequence[tuple[Action, Happening]]],
        after: Callable[[State, Action], State | None],
        couplings: Sequence[object],
        grammar: Grammar,
        surely: float = 0.9,
    ) -> Following:
        """That game followed as far as the notation and the rules agree.

        `happenings` says what could happen at a position — each action the rules leave standing with what it
        would do — and `after` takes one of them. A position where nothing could happen ends the following,
        which is how a finished game ends rather than a failure.

        It stops at the first notation it cannot read down to one move, and says why. Reading on past that
        would be following a different game: every position after it rests on a move that was guessed."""
        state = start
        positions: list[State] = [start]
        taken: list[Action] = []
        for at, said in enumerate(notation):
            offered = tuple(happenings(state))
            if not offered:
                return Following(tuple(positions), tuple(taken), at, "" if at == len(notation) else NOWHERE)
            found = self._decoder.read(
                state, said, couplings, [one for _, one in offered], grammar, surely  # type: ignore[arg-type]
            )
            chosen = self._one(offered, found)
            if chosen is None:
                why = REFUSED if not found else SEVERAL
                logger.info("Followed %d of %d: %r, and %s", at, len(notation), said, why)
                return Following(tuple(positions), tuple(taken), at, why)
            following = after(state, chosen)
            if following is None:
                return Following(tuple(positions), tuple(taken), at, NOWHERE)
            state = following
            positions.append(state)
            taken.append(chosen)
        logger.info("Followed a game of %d moves written by somebody else, whole", len(taken))
        return Following(tuple(positions), tuple(taken), len(notation))

    def _one(
        self, offered: Sequence[tuple[Action, Happening]], found: Sequence[Happening]
    ) -> Action | None:
        """The one action those happenings name, or None where they name no action or several.

        Several happenings may be one action — a move drawn twice over — so what is counted is the actions
        they belong to and not the happenings themselves."""
        held = {action for action, happening in offered if happening in found}
        return next(iter(held)) if len(held) == 1 else None
