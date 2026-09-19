from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar

#: What a state is made of: named data models, each with OMF's methods.
type DataModel = Scalar | List | Grid | Map
