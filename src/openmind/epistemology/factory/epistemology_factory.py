from collections.abc import Mapping

from openmind.inference.model.error_model import ErrorModel
from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.inference.service.bayesian_certainty import BayesianCertainty
from openmind.inference.service.certainty_assessor import CertaintyAssessor
from openmind.inference.service.coherence_checker import CoherenceChecker
from openmind.epistemology.service.epistemology import Epistemology
from openmind.epistemology.service.foundherentism import Foundherentism
from openmind.inference.service.fuzzy_certainty import FuzzyCertainty
from openmind.inference.service.gaussian_certainty import GaussianCertainty
from openmind.inference.service.justifier import Justifier


def create_epistemology(
    error_models: Mapping[str, ErrorModel] | None = None, immediate: Mapping[str, bool] | None = None
) -> Epistemology:
    """Epistemology with its certainty models in the order they are tried — Bayesian, Gaussian, then fuzzy — the error
    models mechanisms have by id, and which contexts are reviewed as soon as a belief is set.

    The tracing, the assessing and the conflict-finding are the engine's; what is handed to them from here is the
    doctrine — what counts as an anchor."""
    justifier = Justifier(Foundherentism())
    accuracy = AccuracyScorer()
    assessor = CertaintyAssessor((BayesianCertainty(accuracy, error_models), GaussianCertainty(accuracy), FuzzyCertainty()))
    return Epistemology(justifier, assessor, CoherenceChecker(justifier), immediate)
