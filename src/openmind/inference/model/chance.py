from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chance:
    """How likely something is, and how that was arrived at.

    `spread` says how much the counting behind it supports it, so a rule fitted on six examples is not mistaken
    for one fitted on six thousand. A number without it invites being read as more settled than it is.

    `exact` says whether the proofs were counted or sampled. Where a conclusion has many proofs leaning on each
    other, counting them exactly can cost more than the answer is worth, and sampling is the honest fallback —
    honest because it says so rather than passing itself off as a count."""

    value: float
    spread: float = 0.0
    derivations: int = 0
    exact: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"A chance of {self.value} is not a chance")

    @property
    def readable(self) -> str:
        said = f"{self.value:.3g}"
        if self.spread:
            said += f" give or take {self.spread:.2g}"
        return said if self.exact else f"about {said}"
