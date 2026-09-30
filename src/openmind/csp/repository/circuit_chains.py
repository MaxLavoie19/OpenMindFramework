class CircuitChains:
    """The runs of positions a circuit's decided edges have joined up, and the trail of how they were joined.

    Every position starts as a chain of one, itself. Deciding that `i` is followed by `j` joins the chain ending at
    `i` to the chain starting at `j`. What this is for is the one thing all-different cannot say: a chain that does
    not yet cover every position must not be closed into a loop, so its own start is refused at its end until it
    does.

    Working storage for one search, handed to the propagator per call, and unwound with the domains it was narrowed
    alongside."""

    __slots__ = ("_positions", "_start", "_end", "_length", "_recorded", "_trail")

    def __init__(self, positions: int) -> None:
        self._positions = positions
        self._start = list(range(positions))
        self._end = list(range(positions))
        self._length = [1] * positions
        self._recorded: dict[int, int] = {}
        self._trail: list[tuple[int, int, int | None]] = []

    @property
    def height(self) -> int:
        return len(self._trail)

    def recorded(self, position: int) -> int | None:
        """What this position was decided to be followed by, or None while nothing has been."""
        return self._recorded.get(position)

    def start_of(self, position: int) -> int:
        """Where the chain ending at this position begins."""
        return self._start[position]

    def end_of(self, position: int) -> int:
        """Where the chain beginning at this position ends."""
        return self._end[position]

    def length_of(self, start: int) -> int:
        """How many positions the chain beginning there covers."""
        return self._length[start]

    def mark(self, position: int, successor: int) -> None:
        """Records the edge without joining anything, which is what closing the last one does."""
        self._trail.append((_RECORDED, position, self._recorded.get(position)))
        self._recorded[position] = successor

    def join(self, position: int, successor: int) -> tuple[int, int, int]:
        """Joins the chain ending at the position to the chain starting at the successor, and gives back the joined
        chain's start, end and length."""
        start = self._start[position]
        end = self._end[successor]
        length = self._length[start] + self._length[successor]
        self._trail.append((_END, start, self._end[start]))
        self._trail.append((_START, end, self._start[end]))
        self._trail.append((_LENGTH, start, self._length[start]))
        self._end[start] = end
        self._start[end] = start
        self._length[start] = length
        return start, end, length

    def undo_to(self, height: int) -> tuple[str, ...]:
        """Puts back everything recorded since that height, most recent first. It changes no domain, so it names no
        variable."""
        while len(self._trail) > height:
            kind, index, held = self._trail.pop()
            if kind == _RECORDED:
                if held is None:
                    self._recorded.pop(index, None)
                else:
                    self._recorded[index] = held
            elif kind == _START:
                self._start[index] = held  # type: ignore[assignment]
            elif kind == _END:
                self._end[index] = held  # type: ignore[assignment]
            else:
                self._length[index] = held  # type: ignore[assignment]
        return ()


_RECORDED = 0
_START = 1
_END = 2
_LENGTH = 3
