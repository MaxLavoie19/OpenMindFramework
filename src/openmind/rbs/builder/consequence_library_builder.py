from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.inference.service.mechanics import Mechanics
from openmind.inference.service.memory_meter import MemoryMeter
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class ConsequenceLibraryBuilder:
    """Wires a consequence library with its own solver, predictor, state reader and variable name mapper, and mechanics
    sharing its solver and predictor, with a memory meter."""

    def build(self) -> ConsequenceLibrary:
        solver, predictor, names = SolverBuilder().build(), PredictorBuilder().build(), VariableNameMapper()
        mechanics = Mechanics(solver, predictor, StateNamespaceMapper(names), MemoryMeter())
        return ConsequenceLibrary(solver, predictor, StateReader(), names, mechanics)
