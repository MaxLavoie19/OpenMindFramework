from dataclasses import dataclass

from openmind.dashboard.model.log_progress import LogProgress
from openmind.dashboard.model.machine_status import MachineStatus
from openmind.dashboard.model.report_summary import ReportSummary
from openmind.dashboard.model.round_games import RoundGames


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """Everything the page shows at one moment: the domain, when the snapshot was taken, the newest training report and
    training log (None when there are none), the machine, and what each round's self-play games came to, which the
    plots are drawn from."""

    domain: str
    taken_at: str
    report: ReportSummary | None
    progress: LogProgress | None
    machine: MachineStatus
    played: tuple[RoundGames, ...] = ()
