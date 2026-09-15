from pathlib import Path


class IncrementalLineReader:
    """Reads a growing file a piece at a time: each call gives the complete lines written since the previous call for
    that file, so a log of hundreds of megabytes is read once, then only what's new. A line not ended yet waits for the
    next call; a file that got shorter, or was replaced, is read again from its start."""

    def __init__(self) -> None:
        self._offsets: dict[Path, tuple[int, int]] = {}

    def new_lines(self, path: Path) -> list[str]:
        """The complete lines added since the previous call, in order; a missing file gives none."""
        try:
            status = path.stat()
        except FileNotFoundError:
            self._offsets.pop(path, None)
            return []
        inode, offset = self._offsets.get(path, (status.st_ino, 0))
        if inode != status.st_ino or status.st_size < offset:
            offset = 0
        with path.open("rb") as file:
            file.seek(offset)
            data = file.read()
        end = data.rfind(b"\n") + 1
        self._offsets[path] = (status.st_ino, offset + end)
        return data[:end].decode("utf-8", errors="replace").splitlines()

    def forget(self, path: Path) -> None:
        """Reads the file from its start at the next call."""
        self._offsets.pop(path, None)
