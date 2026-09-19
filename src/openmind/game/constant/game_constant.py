#: The entry points a project registers its games under: each maps a game's name to a function declaring it,
#: `declare(name, knowledge_base) -> context name`, `name` being the game or "game/variant".
DOMAIN_ENTRY_POINTS = "openmind.domains"

#: What separates a game's name from its variant's: "tictactoe/fourinarow".
VARIANT_SEPARATOR = "/"
