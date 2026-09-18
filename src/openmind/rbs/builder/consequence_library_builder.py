from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class ConsequenceLibraryBuilder:
    """Wires a consequence library with its state reader and variable name mapper, and mechanics with a memory meter.
    What a rule reads about a position's consequences comes from the RBS it is given, so neither needs a solver or a
    predictor of its own."""

    def build(self) -> ConsequenceLibrary:
        names = VariableNameMapper()
        mechanics = Mechanics(StateNamespaceMapper(names), MemoryMeter())
        return ConsequenceLibrary(StateReader(), names, mechanics)
