from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    """Where the dashboard reads a domain's training from: the domain, the directory of its training reports and of its
    training logs (each holding `<domain>/` below it), the system log earlyoom writes to (None to skip it), and the
    process file system."""

    domain: str
    report_directory: Path
    log_directory: Path
    syslog: Path | None
    proc: Path = Path("/proc")
