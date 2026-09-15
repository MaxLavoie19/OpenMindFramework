import os


class MemoryMeter:
    """How much memory this process holds, as the operating system counts it."""

    def resident_bytes(self) -> int:
        """The resident memory, from /proc/self/statm where the system has it; otherwise the peak so far."""
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            import resource  # Only where /proc is missing; Windows has neither.

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
