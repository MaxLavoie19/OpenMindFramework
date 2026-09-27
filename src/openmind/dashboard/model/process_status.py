from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessStatus:
    """A process worth showing: its id, its role (loop, run, training or worker), the memory it holds in bytes, how
    long it has run in seconds, and its command line.

    `run` is which run it belongs to, for a process a run said was its own or that descends from one, and empty for
    a process recognised by its command line instead. A name a run gave itself beats a role guessed from a string,
    which is the whole reason it is here."""

    pid: int
    role: str
    rss_bytes: int
    seconds: float
    command: str
    run: str = ""
