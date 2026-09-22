from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceBudget:
    """What one question may spend.

    Every limit here is the caller's. The engine sets none of its own, because a limit it chose would be a guess
    about how hard the question is, and it has no way to make that guess: the same question is trivial in one game
    and unreachable in another. What follows from a set of rules is endless, so something must say when to stop —
    and what should say it is whoever knows how much the answer is worth.

    `derivations` is how many proofs to collect before stopping. It is not one: a chance cannot be worked out from
    a single proof, and neither can whether a conclusion has more than one reason behind it. `nodes` is how large
    the decision diagram may grow before a chance is sampled rather than counted exactly, and `draws` how many
    times the clauses are drawn when it is — as many as the nodes allowed, where the caller says nothing, since
    spending the same budget either way is the one answer that invents no number of its own."""

    seconds: float
    steps: int | None = None
    depth: int | None = None
    derivations: int | None = None
    nodes: int = 100_000
    draws: int | None = None

    @property
    def drawing(self) -> int:
        """How many draws a sampled chance is taken over."""
        return self.nodes if self.draws is None else self.draws

    def __post_init__(self) -> None:
        if self.seconds <= 0.0:
            raise ValueError("A question needs some time to answer it")
        for name in ("steps", "depth", "derivations", "draws"):
            held = getattr(self, name)
            if held is not None and held < 1:
                raise ValueError(f"A budget of {held} {name} leaves nothing to do")
        if self.nodes < 1:
            raise ValueError("A budget of no nodes leaves no way to work a chance out")
