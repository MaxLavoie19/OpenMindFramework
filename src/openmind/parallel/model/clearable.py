from typing import Protocol


class Clearable(Protocol):
    """A cache a process's memory guard can empty: how many entries it keeps, and forgetting them all."""

    def memory_entries(self) -> int: ...

    def clear_memory(self) -> None: ...
