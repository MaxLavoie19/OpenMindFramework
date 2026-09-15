from dataclasses import dataclass
from pathlib import Path

from openmind.dashboard.model.round_row import RoundRow


@dataclass(frozen=True, slots=True)
class ReportSummary:
    """A training report as the page shows it: the report, when the training started, whether it's complete, every
    round, the latest round's value rules as (term, weight), the largest weight first, and the signals the latest round
    followed as (name, agreements, disagreements, accuracy, reliability, games, wins, draws, losses)."""

    path: Path
    created_at: str
    complete: bool
    rounds: tuple[RoundRow, ...]
    latest_rules: tuple[tuple[str, float], ...]
    latest_arms: tuple[tuple[str, int, int, float, float, int, int, int, int], ...] = ()
