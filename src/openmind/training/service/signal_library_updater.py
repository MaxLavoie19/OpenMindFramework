import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from openmind.rbs.model.value_generation_result import ValueGenerationResult
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.rule_support import RuleSupport
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord

logger = logging.getLogger(__name__)


class SignalLibraryUpdater:
    """Updates the rules a library's signals support after a round's fits, one fit per signal:

    - a rule's strength from a signal is its standardized weight in the fit chosen for that signal's targets; a signal
      fitted this round replaces its earlier strengths, so a rule its new fit doesn't weight loses its support, and
      strengths from signals not fitted this round stay;
    - a rule's standing is the sum, over the signals supporting it, of the signal's reliability times the strength's
      size; a signal never read yet counts as fully reliable, nothing being known against it, so every signal starts
      the same;
    - a rule whose standing is 0, supported by no reliable signal, leaves the library. There is no limit on how many
      stay; the rules are kept highest standing first;
    - the value bases become this round's fits, one per signal fitted: the arms the next round's agents follow."""

    def update(self, library: SignalLibrary, fits: Mapping[str, ValueGenerationResult]) -> SignalLibrary:
        merged: dict[PythonRule, dict[str, float]] = {}
        for support in library.supports:
            staying = {name: strength for name, strength in support.strengths if name not in fits}
            if staying:
                merged[support.term] = staying
        for name, result in fits.items():
            for term, strength in result.strengths:
                if strength != 0.0:
                    merged.setdefault(term, {})[name] = strength
        ranked: list[tuple[float, RuleSupport]] = []
        for term, strengths in merged.items():
            support = RuleSupport(term, tuple(sorted(strengths.items())))
            standing = self.standing(library, support)
            if standing > 0.0:
                ranked.append((standing, support))
        staying = {support.term for _, support in ranked}
        gone = [term for term in dict.fromkeys((*(support.term for support in library.supports), *merged)) if term not in staying]
        dropped = len(gone)
        for term in gone:
            logger.debug("Dropped rule %s: no reliable signal supports it", term.source)
        ranked.sort(key=lambda item: -item[0])
        for standing, support in ranked:
            logger.debug(
                "Rule %s: standing %s; %s",
                support.term.source,
                standing,
                " ".join(f"{name}={strength}" for name, strength in support.strengths),
            )
        logger.info(
            "Signals support %d rules after fitting %d signals; dropped %d rules no reliable signal supports",
            len(ranked),
            len(fits),
            dropped,
        )
        return SignalLibrary(
            library.domain,
            library.records,
            tuple(support for _, support in ranked),
            tuple((name, result.value_base) for name, result in fits.items()),
        )

    def score(self, library: SignalLibrary, games: Sequence[PlayedGame]) -> SignalLibrary:
        """The library with every game between arms added to its arms' records: a win, a draw or a loss each."""
        records = {record.signal.name: record for record in library.records}
        for game in games:
            if len(game.arms) != 2:
                continue
            first, second = game.payoffs[0], game.payoffs[1]
            outcomes = ("draws", "draws") if first == second else (("wins", "losses") if first > second else ("losses", "wins"))
            for arm, outcome in zip(game.arms, outcomes, strict=True):
                record = records.get(arm, SignalRecord(Signal(arm)))
                records[arm] = replace(record, games=record.games + 1, **{outcome: getattr(record, outcome) + 1})
        for arm in dict.fromkeys(arm for game in games for arm in game.arms):
            record = records[arm]
            logger.info(
                "Arm %s: %d wins, %d draws, %d losses of %d games, score %s",
                arm,
                record.wins,
                record.draws,
                record.losses,
                record.games,
                (record.wins + record.draws / 2) / record.games if record.games else None,
            )
        return replace(library, records=tuple(records.values()))

    def standing(self, library: SignalLibrary, support: RuleSupport) -> float:
        """Σ reliability × |strength| over the signals supporting the rule."""
        records = {record.signal.name: record for record in library.records}
        return math.fsum(self._reliability(records.get(name)) * abs(strength) for name, strength in support.strengths)

    def _reliability(self, record: SignalRecord | None) -> float:
        if record is None or record.agreements + record.disagreements == 0:
            return 1.0
        return record.reliability
