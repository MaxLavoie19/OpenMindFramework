from dataclasses import dataclass

from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class Relaxation:
    """Fewer rules, to see what a game teaches without them: which constraints to drop, by action and position; which
    parameters take wider values, by action and parameter, with the rule giving them; and whether every player also has a
    `pass` action, which only hands the turn over. What a relaxed domain teaches is a hypothesis, kept only when it helps
    in real games."""

    name: str
    dropped: tuple[tuple[str, int], ...] = ()
    widened: tuple[tuple[str, str, Rule], ...] = ()
    passing: bool = False
