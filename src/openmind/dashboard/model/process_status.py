from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessStatus:
    """A training process: its id, its role (loop, training or worker), the memory it holds in bytes, how long it has
    run in seconds, and its command line."""

    pid: int
    role: str
    rss_bytes: int
    seconds: float
    command: str
