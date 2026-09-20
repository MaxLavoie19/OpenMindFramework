from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """How much a planner may explore, and how. `nodes` is how many nodes it may expand, None for as many as it needs;
    `depth` how far it may look, None for as far as the game goes; `seed` what its random choices follow.

    `exploration` is how much a Monte-Carlo tree search weighs what a move was rated against what its visits found,
    and `regret_exploration` how much of a choice stays uniform where several players act at once, so a move nothing
    regrets yet is still tried.

    `temperature` is how a search turns what it explored into what it plays. At 1 the strategy is what it explored, so
    a move looked at half as often is played half as often: that is what bootstrapping and self-play want, since a
    game only teaches what it was allowed to try. At 0 it plays the best move it found and nothing else, which is what
    competitive play wants. In between it sharpens what it explored without throwing the rest away.

    How much to spend is the time management policy's; the settings are what it sets, rather than constants in the
    code, so what each one brings can be learned."""

    nodes: int | None = None
    depth: int | None = None
    seed: int = 0
    exploration: float = 1.5
    regret_exploration: float = 0.6
    temperature: float = 1.0
