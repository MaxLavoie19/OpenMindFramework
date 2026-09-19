from collections.abc import Mapping

from openmind.epistemology.model.error_model import ErrorModel
from openmind.epistemology.service.accuracy_scorer import AccuracyScorer
from openmind.epistemology.service.bayesian_certainty import BayesianCertainty
from openmind.epistemology.service.certainty_assessor import CertaintyAssessor
from openmind.epistemology.service.coherence_checker import CoherenceChecker
from openmind.epistemology.service.epistemology import Epistemology
from openmind.epistemology.service.fuzzy_certainty import FuzzyCertainty
from openmind.epistemology.service.gaussian_certainty import GaussianCertainty
from openmind.epistemology.service.justifier import Justifier


def create_epistemology(
    error_models: Mapping[str, ErrorModel] | None = None, immediate: Mapping[str, bool] | None = None
) -> Epistemology:
    """Epistemology with its certainty models in the order they are tried — Bayesian, Gaussian, then fuzzy — the error
    models mechanisms have by id, and which contexts are reviewed as soon as a belief is set."""
    justifier = Justifier()
    accuracy = AccuracyScorer()
    assessor = CertaintyAssessor((BayesianCertainty(accuracy, error_models), GaussianCertainty(accuracy), FuzzyCertainty()))
    return Epistemology(justifier, assessor, CoherenceChecker(justifier), immediate)
