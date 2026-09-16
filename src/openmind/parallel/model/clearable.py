from typing import Protocol


class Clearable(Protocol):
    """A cache a process's memory guard can bound: how many entries it keeps, keeping only the newest so many of them,
    and forgetting them all.

    `evict_memory` is what the guard asks for as a matter of course, since a cache emptied wholesale loses the working
    set a search keeps coming back to; `clear_memory` is for when a process must give back whatever it can."""

    def memory_entries(self) -> int: ...

    def evict_memory(self, entries: int) -> None: ...

    def clear_memory(self) -> None: ...
