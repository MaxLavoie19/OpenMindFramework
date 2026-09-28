from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Choice:
    """One decision somebody faced: what they could have done, and what they did.

    `candidates` is a row per thing they could have done and a column per term, so the row at `taken` is the
    one they took. **What is known is the ranking and not a value**: nobody said what the move was worth to
    them, only that they preferred it to the others in front of them at that moment.

    That is why the others matter. A move taken from three candidates says little; the same move taken from
    forty says a great deal, and a fit that saw only what was played could not tell the two apart. The
    candidates are the evidence as much as the choice is."""

    candidates: np.ndarray
    taken: int

    @property
    def offered(self) -> int:
        """How many things were on offer. One is a decision nobody made."""
        return int(self.candidates.shape[0])

    @property
    def decides(self) -> bool:
        """Whether anything was actually chosen between."""
        return self.offered > 1 and 0 <= self.taken < self.offered
