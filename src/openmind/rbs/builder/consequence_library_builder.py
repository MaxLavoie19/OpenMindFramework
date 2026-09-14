from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class ConsequenceLibraryBuilder:
    """Wires a consequence library with its own solver, predictor, state reader and variable name mapper."""

    def build(self) -> ConsequenceLibrary:
        return ConsequenceLibrary(SolverBuilder().build(), PredictorBuilder().build(), StateReader(), VariableNameMapper())
