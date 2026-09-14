from collections.abc import Callable

from openmind.agent.model.domain import Domain

#: What an installed project registers under the openmind.domains entry points: a function giving its domain from the
#: whole domain name, such as "chess" or "chess/960", and raising ValueError for a name it doesn't know.
type DomainRecipe = Callable[[str], Domain]
