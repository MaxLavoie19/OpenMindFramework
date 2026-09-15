from datetime import datetime

from openmind.dashboard.model.dashboard_settings import DashboardSettings
from openmind.dashboard.model.dashboard_snapshot import DashboardSnapshot
from openmind.dashboard.service.log_progress_reader import LogProgressReader
from openmind.dashboard.service.machine_reader import MachineReader
from openmind.dashboard.service.report_reader import ReportReader


class DashboardService:
    """Takes snapshots of a domain's training: its newest report, its newest log's progress, and the machine. Keeps its
    readers' places between snapshots, so each snapshot reads only what's new."""

    def __init__(self, report_reader: ReportReader, log_progress_reader: LogProgressReader, machine_reader: MachineReader) -> None:
        self._report_reader = report_reader
        self._log_progress_reader = log_progress_reader
        self._machine_reader = machine_reader

    def snapshot(self, settings: DashboardSettings) -> DashboardSnapshot:
        return DashboardSnapshot(
            settings.domain,
            datetime.now().replace(microsecond=0).isoformat(sep=" "),
            self._report_reader.summary(settings.report_directory / settings.domain),
            self._log_progress_reader.progress(settings.log_directory / settings.domain),
            self._machine_reader.status(settings.proc, settings.syslog),
        )
