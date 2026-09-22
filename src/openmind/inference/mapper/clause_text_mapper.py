from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Functor, Number, Term, Variable

#: How a clause is said in words. A rule reads as its conclusion first, then what it asks for.
RULE = "{head} where {body}"
FACT = "{head}"
GOAL = "is there {body}?"
CONTRADICTION = "nothing can hold"

#: How often it holds, where it does not always.
SOMETIMES = "{said}, {share:g} of the time"

#: How a denial reads.
DENIED = "not {said}"

#: How the things a literal is said of are joined on to it.
ABOUT = "{predicate} {arguments}"


class ClauseTextMapper:
    """A clause in words.

    `Clause.readable` writes a clause the way logic is written, which is right for a log line read by whoever wrote
    the engine and wrong for everything else. This writes it as a sentence.

    Two things want that. A person looking at what OMF concluded should not have to read normal forms to find out
    whether it is sensible — an explanation nobody can read explains nothing. And a clause on its way out into
    natural language has to start somewhere, and starting from a sentence-shaped rendering leaves a language model
    far less to invent than starting from punctuation.

    It reads predicates as the phrases they are. The readings a game offers are already named in words — a
    predicate is a reading's name with its slots opened up — so saying the predicate and then what it was said of
    gives a sentence without anything having to be looked up. Nothing here knows what any particular predicate
    means, and nothing here is any one game's."""

    def to_text(self, clause: Clause) -> str:
        """The clause as a sentence."""
        said = self._said(clause)
        return said if clause.certain else SOMETIMES.format(said=said, share=clause.probability)

    def literal_to_text(self, literal: Literal) -> str:
        """One thing said, in words."""
        said = ABOUT.format(
            predicate=literal.predicate, arguments=self._joined(literal.arguments)
        ).strip() if literal.arguments else literal.predicate
        return DENIED.format(said=said) if literal.negated else said

    def term_to_text(self, term: Term) -> str:
        """One thing said of, in words."""
        if isinstance(term, Variable):
            return f"any {term.sort or term.name.split('#')[0].lower()}"
        if isinstance(term, Number):
            return f"{term.value:g}" if isinstance(term.value, float) else str(term.value)
        if isinstance(term, Functor):
            if not term.arguments:
                return f"some {term.name.strip('#').replace('invented', 'thing ')}".strip()
            return f"the {term.name} of {self._joined(term.arguments)}"
        return "nothing" if term.name is None else str(term.name)

    def _said(self, clause: Clause) -> str:
        if clause.empty:
            return CONTRADICTION
        head, body = clause.head, clause.body
        if head is None:
            return GOAL.format(body=self._joined_literals(clause.body))
        if not body:
            return FACT.format(head=self.literal_to_text(head))
        return RULE.format(head=self.literal_to_text(head), body=self._joined_literals(body))

    def _joined_literals(self, literals: tuple[Literal, ...]) -> str:
        said = [self.literal_to_text(one) for one in literals]
        if len(said) < 2:
            return "".join(said)
        return ", ".join(said[:-1]) + " and " + said[-1]

    def _joined(self, terms: tuple[Term, ...]) -> str:
        said = [self.term_to_text(one) for one in terms]
        if len(said) < 2:
            return "".join(said)
        return ", ".join(said[:-1]) + " and " + said[-1]
