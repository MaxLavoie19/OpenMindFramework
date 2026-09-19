from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelMeasure:
    """What a model has been measured at: how often it turned out right (`accuracy`), how far off it is on numbers
    (`spread`), the mean seconds a reading takes (`processing_seconds`) and how many readings the timing rests on.
    Anything not measured yet is None."""

    accuracy: float | None = None
    spread: float | None = None
    processing_seconds: float | None = None
    readings: int = 0
