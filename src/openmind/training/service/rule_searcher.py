import logging
from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.inference.model.deduction import Deduction
from openmind.inference.model.expression import Expression
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.training.constant.signal_constant import UNIFORM, WEIGHTED, WIN
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.service.signal_library_updater import SignalLibraryUpdater
from openmind.training.service.signal_ranker import SignalRanker
from openmind.training.service.signal_targeter import SignalTargeter

logger = logging.getLogger(__name__)


class RuleSearcher:
    """Searches for new rules after a decisive game, over every game played so far: the signals followed, winning, the
    best recorded signals and their uniform and weighted aggregations, get their targets on every position of every
    game for each player, proven positions at their proven payoffs; the expression search, starting from the seeds
    proofs induced, fits value rules to each within its budget; and the library's rules and value bases become the
    fits', the arms the next games follow."""

    def __init__(
        self,
        signal_targeter: SignalTargeter,
        signal_ranker: SignalRanker,
        value_generator: ValueGenerator,
        signal_library_updater: SignalLibraryUpdater,
    ) -> None:
        self._signal_targeter = signal_targeter
        self._signal_ranker = signal_ranker
        self._value_generator = value_generator
        self._signal_library_updater = signal_library_updater

    def search(
        self,
        domain: Domain,
        library: SignalLibrary,
        games: Sequence[PlayedGame],
        deductions: Sequence[Deduction],
        seeds: Sequence[Expression],
        settings: ContinuousTrainingSettings,
    ) -> SignalLibrary:
        followed = (Signal(WIN), *(record.signal for record in self._signal_ranker.best(library, settings.signals.arms)))
        parts = tuple(signal.name for signal in followed)
        signals = (*followed, Signal(UNIFORM, parts=parts), Signal(WEIGHTED, parts=parts))
        reliabilities = self._signal_ranker.reliabilities(library, followed)
        rows = self._signal_targeter.rows(domain, games)
        if not rows:
            return library
        targets = self._signal_targeter.targets(domain, games, signals, reliabilities, settings.signals.horizon, deductions)
        empty = np.zeros(0)
        generations = self._value_generator.generate_for_targets(
            domain, rows, (), {name: (values, empty) for name, values in targets.items()}, settings.values, seeds
        )
        updated = self._signal_library_updater.update(library, generations)
        logger.info(
            "Searched rules on %d games and %d positions for %d signals: %s",
            len(games),
            len(rows) // len(domain.players.names),
            len(generations),
            "; ".join(f"{name} {len(result.value_base.rules)} rules" for name, result in generations.items()),
        )
        return updated
