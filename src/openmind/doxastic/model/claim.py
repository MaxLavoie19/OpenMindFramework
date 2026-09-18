from dataclasses import dataclass

from openmind.doxastic.constant.doxastic_constant import CHAIN_SEPARATOR, KEY_SEPARATOR, SUBJECT_SEPARATOR
from openmind.rbs.model.rule import Rule


@dataclass(frozen=True, slots=True)
class Claim:
    """Something that can be true or false, which the agent can hold evidence for and against.

    `rule` reads it on a position: True, False, or None where it can't be read there; a claim without one is a claim
    nothing can check yet, such as something the agent was told. `about` is who or what the claim concerns. `holder` is
    whose belief it is, as a chain: `()` the agent's own, `("black",)` what the agent believes black believes,
    `("black", "white")` what the agent believes black believes white believes."""

    name: str
    rule: Rule | None = None
    about: tuple[str, ...] = ()
    holder: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """What identifies the claim: its name, whose belief it is and what it is about. Two claims with the same key
        are the same claim, whether or not both carry the rule reading it."""
        return KEY_SEPARATOR.join(
            (self.name, CHAIN_SEPARATOR.join(self.holder), SUBJECT_SEPARATOR.join(sorted(self.about)))
        )

    @property
    def believed_by(self) -> str | None:
        """Whose belief the claim is, the last of the chain, or None when it is the agent's own."""
        return self.holder[-1] if self.holder else None
