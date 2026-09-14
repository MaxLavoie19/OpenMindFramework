from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """How hypotheses are generated and validated. Samples need min_visits visits. A hypothesis combines up to
    max_conditions conditions; the beam_width best of each size are kept and extended. Discovery keeps a hypothesis
    whose matching actions have min_rule_visits visits and whose effect on advantage is at least min_gain; priority
    rules are judged at the given confidence. near() offsets reach max_offset from the variable an action sets;
    solo_distance() looks solo_limit own moves ahead; up to patterns winning moves are probed for goal patterns.
    Validation runs a permutation test with this many permutations and keeps hypotheses at the false_discovery_rate.
    With coverage, a validated rule a simpler rule covers is left out."""

    min_visits: int
    max_conditions: int
    min_rule_visits: int
    min_gain: float
    confidence: float
    beam_width: int
    max_offset: int
    solo_limit: int
    patterns: int
    false_discovery_rate: float
    permutations: int
    coverage: bool = True
