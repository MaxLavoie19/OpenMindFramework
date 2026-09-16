from dataclasses import dataclass
from pathlib import Path

from openmind.doxastic.constant.doxastic_constant import KNOWLEDGE_DIRECTORY


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    """Where the dashboard reads a domain's training from: the domain, the directory of its training reports and of its
    training logs (each holding `<domain>/` below it), the system log earlyoom writes to (None to skip it), the process
    file system, and the knowledge directory the games are remembered in."""

    domain: str
    report_directory: Path
    log_directory: Path
    syslog: Path | None
    proc: Path = Path("/proc")
    knowledge_directory: Path = Path(KNOWLEDGE_DIRECTORY)
