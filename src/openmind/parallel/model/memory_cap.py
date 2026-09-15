from dataclasses import dataclass
from pathlib import Path

from openmind.parallel.constant.parallel_constant import MEMORY_GRACE_SECONDS


@dataclass(frozen=True, slots=True)
class MemoryCap:
    """How many bytes each worker process holds at most, where a worker ended for staying over that writes its diagnosis,
    and how long a worker may stay over the cap once it asked its caches to clear."""

    worker_bytes: int
    diagnosis_directory: Path
    grace_seconds: float = MEMORY_GRACE_SECONDS

    def __post_init__(self) -> None:
        if self.worker_bytes < 1:
            raise ValueError(f"A memory cap needs at least 1 byte per worker, not {self.worker_bytes}")
        if self.grace_seconds < 0.0:
            raise ValueError(f"A memory cap's grace can't be negative, not {self.grace_seconds}")
