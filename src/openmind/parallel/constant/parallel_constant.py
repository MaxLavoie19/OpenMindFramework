import os


def _half_the_memory() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    except (AttributeError, OSError, ValueError):
        return 8 * 1024**3


#: Half the logical CPUs: searches are memory hungry, and hyperthreads add little to them.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)
#: Slices per worker when a list is split: enough to balance uneven slices, few enough that each slice's setup, such
#: as building an agent, stays small.
SLICES_PER_WORKER = 4
#: How worker processes start: a fresh interpreter, the same on every platform.
START_METHOD = "spawn"
#: How often a worker checks that the process that started it is still its parent, in seconds.
PARENT_CHECK_SECONDS = 5.0
#: How long a worker gets to end once it is asked to, in seconds, before it is terminated.
STOP_SECONDS = 5.0

#: Half the machine's memory, and that half shared between the logical CPUs: how many bytes a process holds before its
#: memory guard clears its caches, until the process is given a limit of its own.
HALF_THE_MEMORY = _half_the_memory()
DEFAULT_PROCESS_MEMORY = HALF_THE_MEMORY // (os.cpu_count() or 1)
#: Entries a process's caches keep between two reads of the process's memory.
MEMORY_CHECK_INTERVAL = 1_000
#: How often a worker under a memory cap reads the memory it holds, in seconds, and how long by default it may stay over
#: the cap once it asked its caches to clear, before it ends itself.
MEMORY_WATCH_SECONDS = 1.0
MEMORY_GRACE_SECONDS = 5.0
#: The share of their entries the caches keep each time a process is found over its limit: the newest stay, the oldest
#: go, and the next reading cuts again if it is still over. A share rather than a size in bytes, because what an entry
#: costs can't be told from what the process holds — most of that is no cache of its own. It is also the share of a
#: worker's memory cap the guard takes as its limit, so the caches are cut back before the cap ends the worker.
MEMORY_SETTLE_SHARE = 0.85
#: Seconds between two log lines of a memory guard about clearing its caches; clears in between are counted.
MEMORY_LOG_SECONDS = 60.0
#: The exit code of a worker that ended itself for staying over its memory cap.
MEMORY_EXIT_CODE = 86
#: How many of a worker's latest calls, and of its most numerous object types, a memory diagnosis lists.
DIAGNOSIS_CALLS = 50
DIAGNOSIS_TYPES = 20
#: How many source lines `openmind-rerun-call` shows by default, those holding the most memory first.
DEFAULT_RERUN_LINES = 20
