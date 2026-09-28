import logging
from collections.abc import Sequence

import numpy as np

from openmind.inference.model.choice import Choice
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class ChoiceReader:
    """Decisions somebody faced, read into the terms a preference is fitted in.

    **The readings are the terms, and they need no thresholds.** The mechanism this follows mines rules of the
    shape *one feature, one threshold, one direction* — `victim_value at least 3` — and has to choose the
    thresholds from the observed distribution, because a fixed grid wastes most of its rules on ranges nothing
    occupies. OMF's readings are already that shape: a reading of a candidate either holds or it does not, so
    a column is a reading and a row is a candidate, and nothing has to be cut anywhere.

    **A reading that never varies within a decision is dropped, and this is arithmetic and not a judgement.**
    The score of a candidate is its terms times their weights, and the choosing is a softmax over the
    candidates of one decision. A column that is the same for every candidate of a decision adds the same
    number to all of them, and the same number added to every candidate cancels — exactly as the bias does.
    So such a column contributes nothing to that decision whatever weight it is given, and a column constant
    within *every* decision contributes nothing at all, provably, however it might combine with others.

    That last clause is the difference between this and a coverage bound. Dropping a term for firing too
    rarely or too often is a guess that it will not earn its place; dropping one that cancels is a fact about
    the arithmetic. This project has already paid once for a filter that sounded like arithmetic and was a
    guess — a column flat across a walk turned out to be anything but flat where it was fitted."""

    def __init__(self, candidate_readings: CandidateReadings | None = None) -> None:
        self._readings = CandidateReadings() if candidate_readings is None else candidate_readings

    def read(
        self, decisions: Sequence[tuple[State, Sequence[Action], int]]
    ) -> tuple[tuple[Choice, ...], tuple[str, ...]]:
        """Those decisions as choices, with what each column is called.

        A decision is a position, the actions that were on offer there, and which of them was taken. Anything
        offering fewer than two actions is left out: nobody chose anything, so there is nothing in it.

        The names come back because a weight that cannot say what it is about is a number, and the whole
        reason for fitting rules rather than a network is that the parameters are sentences."""
        said = [
            (state, tuple(actions), taken)
            for state, actions, taken in decisions
            if len(actions) > 1 and 0 <= taken < len(actions)
        ]
        if not said:
            return (), ()
        read = [
            [frozenset(one.readable for one in self._readings.read(state, action)) for action in actions]
            for state, actions, _ in said
        ]
        columns = self._varying(read)
        if not columns:
            logger.info("Nothing to fit a preference on: no reading told one candidate from another")
            return (), ()
        at = {name: index for index, name in enumerate(columns)}
        choices = tuple(
            Choice(
                np.array(
                    [[1.0 if name in held else 0.0 for name in columns] for held in candidates], dtype=float
                ),
                taken,
            )
            for (_, _, taken), candidates in zip(said, read, strict=True)
        )
        logger.info(
            "Read %d decisions of %d candidates on average into %d terms, from %d seen",
            len(choices),
            sum(one.offered for one in choices) // len(choices),
            len(columns),
            len({name for candidates in read for held in candidates for name in held}),
        )
        return choices, columns

    def _varying(self, read: Sequence[Sequence[frozenset[str]]]) -> tuple[str, ...]:
        """The readings that tell one candidate from another somewhere, in a settled order.

        A reading holding for every candidate of a decision, or for none of them, says nothing about that
        decision; one that does so in every decision says nothing anywhere. Sorted so that two runs over the
        same games put the same term in the same column, which is what lets a fitted weight be compared with
        another run's."""
        found: set[str] = set()
        for candidates in read:
            seen = [name for held in candidates for name in held]
            for name in set(seen):
                if 0 < seen.count(name) < len(candidates):
                    found.add(name)
        return tuple(sorted(found))
