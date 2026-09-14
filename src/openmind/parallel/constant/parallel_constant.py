import os

#: Half the logical CPUs: searches are memory hungry, and hyperthreads add little to them.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)
#: Slices per worker when a list is split: enough to balance uneven slices, few enough that each slice's setup, such
#: as building an agent, stays small.
SLICES_PER_WORKER = 4
#: How worker processes start: a fresh interpreter, the same on every platform.
START_METHOD = "spawn"
