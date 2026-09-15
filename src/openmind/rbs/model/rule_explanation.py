from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuleExplanation:
    """A value rule explained: its source and weight; its literal reading; a language model's sentence, None without a
    model or when the model gave none; and that model's name, None without one."""

    source: str
    weight: float
    reading: str
    sentence: str | None
    model: str | None
