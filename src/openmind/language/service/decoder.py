import logging
from collections.abc import Sequence

from openmind.language.model.coupling import Coupling
from openmind.language.model.grammar import Grammar
from openmind.language.model.happening import Happening
from openmind.structure.model.value import Value
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Decoder:
    """What a game's notation says, read as the happenings it could name — and written back, which is the same
    couplings run the other way.

    **One set of rules does both directions.** A coupling says a character in a role speaks of a part of what
    happened. Reading, it narrows the happenings a notation could name; writing, it says which characters a
    happening calls for. Writing is the harder direction and the better test: putting `Nf3` rather than `Ngf3`
    means knowing the origin is recoverable, which is knowing the rules — so a system that writes a game's
    notation has shown the same knowledge as one that reads it, and neither had to be given separately.

    **What is left over is the measure.** A notation that reads as exactly one happening means what is known
    agreed with it. Several means what is known lets through more than the notation assumed. None means what is
    known refuses what the game played. Those are the two errors we score constraints by, arriving from a game
    record rather than from an oracle — so a game nobody has rules for can still be scored.

    It keeps nothing: built once, it is given the position and the notation on every call."""

    def read(
        self,
        state: State,
        said: str,
        couplings: Sequence[Coupling],
        among: Sequence[Happening],
        grammar: Grammar,
        surely: float = 0.9,
    ) -> tuple[Happening, ...]:
        """Which of those happenings the notation could be naming.

        The shape is worked out first, and only the couplings whose role that shape has a place for may speak. A
        notation of no known shape is spoken for by nothing and narrows nothing, which is the honest answer: we
        have not seen its like.

        Only sayings that are right where they apply are allowed to narrow. A saying right four times in five
        refuses the happening that occurred one reading in five, and a handful of those between them refuse every
        candidate there is — which reads as the rules being too tight when the fault is in the reading.

        **Nine tenths and not ninety-nine hundredths, because what the number measures changed.** How sure a
        saying is used to be a bare share, which reaches one from two sightings; it is now counted with a case
        assumed either way, so it never quite reaches one and a symbol seen twenty times tops out near 0.95.
        Measured over twelve hundred moves, nine tenths here names exactly what ninety-nine hundredths named
        before — the same sayings, the same 545 moves written as the game wrote them — while 0.99 under the new
        counting silences six hundred of them. The threshold moved because its meaning did."""
        shape = grammar.shaped(said)
        speaking = [one for one in couplings if one.surely >= surely and one.said_by(shape, said)]
        return tuple(one for one in among if self._agrees(state, one, speaking))

    def write(
        self,
        state: State,
        happening: Happening,
        couplings: Sequence[Coupling],
        grammar: Grammar,
        among: Sequence[Happening] = (),
        surely: float = 0.9,
    ) -> str:
        """The notation for that happening: the briefest saying that still tells it from everything else that
        could have happened.

        **This is where the rules and the notation meet, and it was the last of the three pairings to be
        built.** A game says the least that still identifies what happened — and what counts as *still
        identifying it* is not a fact about the notation. Chess writes `Nf3` where one knight can reach f3 and
        `Nbd2` where two can, and the difference is entirely in what the rules allow. So the choice among the
        shapes that fit is made by reading each back and keeping the shortest that comes to this happening and
        no other.

        `among` is what could have happened — the happenings of the actions the rules leave standing. Given
        none, this falls back to brevity alone, which is what it did before there was anything to ask: the
        shortest shape the happening can fill. That is a stand-in and is now known to be one, because the
        measurement says so — of two thousand moves written that way, nine hundred and thirty-six came out
        differently from the game's own name for them, and disambiguation is what nearly all of them are.

        Where nothing identifies it uniquely, the shortest is given anyway. The notation genuinely cannot say
        it, and saying so by writing something ambiguous is better than saying nothing: the round trip then
        reports "too many", which is the fact."""
        held = happening.said(state)
        found = []
        for shape in grammar.shapes:
            letters = self._filling(shape, held, couplings, surely)
            if letters is not None:
                found.append(letters)
        if not found:
            return ""
        briefest = sorted(found, key=len)
        if not among:
            return briefest[0]
        for letters in briefest:
            # Narrowing to one is what identifying it *means*, and asking instead whether that one is the very
            # happening handed in would almost never say yes: what a move could be told apart from is drawn by
            # the predictor, while the happening being written came from the game, so the two are the same move
            # said by different halves and never the same object. Asked that way this chose the shortest every
            # time, which looked exactly like the rules making no difference — and a measurement said they made
            # none, when in fact a longer saying was the only one that told the move apart a hundred and
            # fifty-four times in twelve hundred.
            if len(self.read(state, letters, couplings, among, grammar, surely)) == 1:
                return letters
        return briefest[0]

    def wanting(
        self, state: State, happening: Happening, couplings: Sequence[Coupling], grammar: Grammar
    ) -> tuple[tuple[str, Value], ...]:
        """Where writing that happening ran out of words, or nothing where it can be written.

        **Failing to write something says more than failing to read it.** A notation that will not read leaves
        too many candidates or none, and either way the count is the whole of the news. A notation that will not
        be *written* points somewhere: this place of this shape speaks of some part, this happening's value for
        that part is such-and-such, and no symbol says it. That is a hole with an address.

        Taken from the shape that got furthest, because the shapes that failed at their first place failed for
        want of everything and name nothing in particular.

        What comes back is the parts that place speaks of, each with what this happening holds for it. A part
        whose value is one this lexicon never learned a symbol for is a word missing; a part the happening has no
        value for at all — `None` — is the model and the notation disagreeing about what there is to say, which
        is the more interesting of the two."""
        held = happening.said(state)
        found: tuple[int, tuple[tuple[str, Value], ...]] = (-1, ())
        for shape in grammar.shapes:
            stopped = self._stopped(shape, held, couplings)
            if stopped is None:
                return ()
            at, wanted = stopped
            if at > found[0]:
                found = (at, wanted)
        return found[1]

    def _stopped(
        self, shape, held: dict, couplings: Sequence[Coupling]
    ) -> tuple[int, tuple[tuple[str, Value], ...]] | None:
        """The first place of that shape nothing can fill and what it wanted, or None where every place fills."""
        for at in range(len(shape)):
            speaking = [one for one in couplings if self._speaks(one, shape, at)]
            if any(held.get(one.part) == one.value for one in speaking):
                continue
            return at, tuple(sorted({(one.part, held.get(one.part)) for one in speaking}, key=str))
        return None

    def _filling(
        self, shape, held: dict, couplings: Sequence[Coupling], surely: float = 0.0
    ) -> str | None:
        """That shape written out, or None where some place of it has nothing to say this happening.

        **Only sayings that are right where they apply may be written, which reading has always insisted on and
        writing did not.** Bits say a symbol is worth attending to; they never say it is always right. Picking
        the most telling symbol regardless of how often it is *correct* writes something informative and wrong:
        measured, it put `c4` where the game wrote `Qe4` — a square the move never went to.

        The two directions have to agree about which sayings count, or the round trip measures the difference
        between them rather than the notation."""
        letters = []
        for at in range(len(shape)):
            saying = [
                one
                for one in couplings
                if one.surely >= surely and self._speaks(one, shape, at) and held.get(one.part) == one.value
            ]
            if not saying:
                return None
            letters.append(max(saying, key=lambda one: one.telling).symbol)
        return "".join(letters)

    def _speaks(self, coupling: Coupling, shape, at: int) -> bool:
        """Whether that saying is one of the sayings of that place of that shape."""
        return coupling.where is not None and any(
            shape == where and at == number for where, number in coupling.where.places
        )

    def _agrees(self, state: State, happening: Happening, speaking: Sequence[Coupling]) -> bool:
        """Whether that happening answers everything the notation's symbols said."""
        said = happening.said(state)
        return all(said.get(one.part) == one.value for one in speaking)

    def scored(
        self,
        state: State,
        said: str,
        couplings: Sequence[Coupling],
        among: Sequence[Happening],
        grammar: Grammar,
        surely: float = 0.9,
    ) -> str:
        """How the reading went, in the words the two errors have here."""
        found = self.read(state, said, couplings, among, grammar, surely)
        if len(found) == 1:
            return "read"
        return "too many" if found else "none"
