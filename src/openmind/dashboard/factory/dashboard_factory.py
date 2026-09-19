from openmind.dashboard.service.dashboard_service import DashboardService
from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader
from openmind.dashboard.service.log_progress_reader import LogProgressReader
from openmind.dashboard.service.machine_reader import MachineReader


def create_dashboard_service() -> DashboardService:
    """A dashboard service whose log and system log readers each keep their own place."""
    return DashboardService(
        LogProgressReader(IncrementalLineReader()), MachineReader(IncrementalLineReader())
    )
