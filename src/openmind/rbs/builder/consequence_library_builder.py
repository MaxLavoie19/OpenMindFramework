from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.world.service.state_reader import StateReader


class ConsequenceLibraryBuilder:
    """Wires a consequence library with its state reader, and mechanics with a memory meter.
    What a rule reads about a position's consequences comes from the RBS it is given, so neither needs a solver or a
    predictor of its own."""

    def build(self) -> ConsequenceLibrary:
        mechanics = Mechanics(StateNamespaceMapper(), MemoryMeter())
        return ConsequenceLibrary(StateReader(), mechanics)
