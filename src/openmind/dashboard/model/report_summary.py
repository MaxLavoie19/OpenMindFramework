from dataclasses import dataclass
from pathlib import Path

from openmind.dashboard.model.round_row import RoundRow


@dataclass(frozen=True, slots=True)
class ReportSummary:
    """A training report as the page shows it: the report, when the training started, whether it's complete, every
    round, and the latest round's value rules as (name, weight), the largest weight first."""

    path: Path
    created_at: str
    complete: bool
    rounds: tuple[RoundRow, ...]
    latest_rules: tuple[tuple[str, float], ...]
