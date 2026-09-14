from openmind.rbs.model.hypothesis_test import HypothesisTest


class HypothesisTextMapper:
    """Maps a tested hypothesis to readable text: place when win_chance(action) >= 1: raises advantage, discovery 0.61
    over 40 states, validation 0.58 over 12 states, p 0.0001, q 0.002, validated."""

    def to_text(self, test: HypothesisTest) -> str:
        conditions = " and ".join(condition.source for condition in test.conditions)
        direction = "raises" if test.direction > 0 else "lowers"
        validation = "none" if test.validation_effect is None else f"{test.validation_effect:.4g}"
        verdict = "validated" if test.validated else "rejected"
        return (
            f"{test.action} when {conditions}: {direction} advantage, discovery {test.discovery_effect:.4g} over "
            f"{test.discovery_states} states, validation {validation} over {test.validation_states} states, "
            f"p {test.p_value:.4g}, q {test.q_value:.4g}, {verdict}"
        )
