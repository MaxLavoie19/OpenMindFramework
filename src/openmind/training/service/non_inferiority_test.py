import numpy as np

from openmind.training.constant.training_constant import BOOTSTRAP_BATCH
from openmind.training.model.non_inferiority import NonInferiority


class NonInferiorityTest:
    """Tests whether one set of choices plays no worse than another by more than a margin, from paired regret
    differences over positions. The percentile bootstrap of the mean difference gives its one-sided upper bound at the
    confidence; the first set is no worse when that bound is below the margin. Differences that are all equal have
    that value as their bound."""

    def test(
        self,
        differences: np.ndarray,
        margin: float,
        confidence: float,
        resamples: int,
        rng: np.random.Generator,
    ) -> NonInferiority:
        count = len(differences)
        if count == 0:
            return NonInferiority(0, 0.0, 0.0, margin, 0.0 < margin)
        mean = float(differences.mean())
        if np.all(differences == differences[0]):
            bound = float(differences[0])
        else:
            means = np.empty(resamples)
            for start in range(0, resamples, BOOTSTRAP_BATCH):
                size = min(BOOTSTRAP_BATCH, resamples - start)
                means[start : start + size] = differences[rng.integers(0, count, size=(size, count))].mean(axis=1)
            bound = float(np.quantile(means, confidence))
        return NonInferiority(count, mean, bound, margin, bound < margin)
