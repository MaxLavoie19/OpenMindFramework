class WorkerEnded(Exception):
    """A call's worker ended abruptly, in a fresh worker too, for a reason other than its memory cap."""
