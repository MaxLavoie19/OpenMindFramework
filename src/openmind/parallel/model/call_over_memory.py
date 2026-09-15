from pathlib import Path


class CallOverMemory(Exception):
    """A call that can't be dropped took a worker over its memory cap in a fresh worker too."""

    def __init__(self, index: int, diagnosis: Path | None) -> None:
        super().__init__(
            f"Call {index} took a worker over its memory cap twice, the second time in a fresh worker; diagnosis {diagnosis}"
        )
        self.index = index
        self.diagnosis = diagnosis

    def __reduce__(self) -> tuple[type, tuple[int, Path | None]]:
        return CallOverMemory, (self.index, self.diagnosis)
