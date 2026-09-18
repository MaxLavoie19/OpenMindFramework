from openmind.inference.model.sort import Sort

#: The sorts every logic knows: truth values, whole numbers and real numbers.
BOOL, INT, REAL = Sort("Bool"), Sort("Int"), Sort("Real")
#: The comparison operators a formula can use between numbers, and the arithmetic operations a term can apply to them.
COMPARISONS = ("<=", "<", ">=", ">")
ARITHMETIC = ("+", "-", "*", "/")
#: The names of the built-in definitions' symbols: set membership and the relations and operations between sets.
MEMBER, SUBSET, UNION, INTERSECTION, DIFFERENCE = "member", "subset", "union", "intersection", "difference"
#: How the symbols of a set selected by a clause, and of the best and worst of a value over a set, are named after
#: what they select or value.
SELECTION_PREFIX, BEST_PREFIX, WORST_PREFIX = "select_", "best_", "worst_"
#: What proving a goal can find: its conclusion follows, its negation follows, neither follows, or it couldn't tell.
PROVED, DISPROVED, INDEPENDENT, UNKNOWN = "proved", "disproved", "independent", "unknown"
#: The proof rule naming a conclusion drawn by induction from its base and step.
INDUCTION_RULE = "induction"
#: How Z3 binds a proof under variables; it concludes nothing of its own, so a proof reads the proof it binds.
PROOF_BIND = "proof-bind"
