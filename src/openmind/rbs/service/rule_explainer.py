import logging
from pathlib import Path

from openmind.agent.model.domain import Domain
from openmind.inference.mapper.expression_sentence_mapper import ExpressionSentenceMapper
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.rbs.constant.explanation_constant import EXPLANATION_PROMPT
from openmind.rbs.model.rule_explanation import RuleExplanation
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.repository.explanation_cache_repository import ExplanationCacheRepository
from openmind.rbs.service.ollama_language_model import OllamaLanguageModel

logger = logging.getLogger(__name__)


class RuleExplainer:
    """Explains a value base's rules: every rule gets its literal reading and, with a language model, a sentence. The
    model is asked with EXPLANATION_PROMPT, which gives it the domain's variables as the initial state holds them, what a
    position's look-aheads mean, and the rule's weight, source and reading. Sentences are kept in the cache by source,
    so a rule is asked about once per model and domain; a rule the model gave no sentence for is asked again next
    time."""

    def __init__(
        self,
        expression_sentence_mapper: ExpressionSentenceMapper,
        expression_generator: ExpressionGenerator,
        explanation_cache_repository: ExplanationCacheRepository,
    ) -> None:
        self._expression_sentence_mapper = expression_sentence_mapper
        self._expression_generator = expression_generator
        self._explanation_cache_repository = explanation_cache_repository

    def explain(
        self,
        value_base: ValueBase,
        domain: Domain,
        language_model: OllamaLanguageModel | None = None,
        cache_directory: Path | None = None,
    ) -> tuple[RuleExplanation, ...]:
        """The rules' explanations, in the value base's order; without a language model, readings only."""
        readings = [self._expression_sentence_mapper.to_sentence(rule.term.source) for rule in value_base.rules]
        if language_model is None:
            return tuple(
                RuleExplanation(rule.term.source, rule.weight, reading, None, None)
                for rule, reading in zip(value_base.rules, readings, strict=True)
            )
        repository = self._explanation_cache_repository
        path = None if cache_directory is None else repository.path(cache_directory, domain.name, language_model.name)
        cache = {} if path is None else repository.load(path)
        variables = self._variables(domain)
        explanations: list[RuleExplanation] = []
        cached = asked = unanswered = 0
        for rule, reading in zip(value_base.rules, readings, strict=True):
            source = rule.term.source
            if source in cache:
                sentence: str | None = cache[source][1]
                cached += 1
            else:
                prompt = EXPLANATION_PROMPT.format(
                    domain=domain.name, variables=variables, weight=rule.weight, source=source, reading=reading
                )
                sentence = language_model.complete(prompt)
                asked += 1
                if sentence is None:
                    unanswered += 1
                else:
                    cache[source] = (reading, sentence)
            explanations.append(RuleExplanation(source, rule.weight, reading, sentence, language_model.name))
        if path is not None and asked > unanswered:
            repository.save(cache, path)
        logger.info(
            "Explained %d %s rules with %s: %d from the cache, %d asked, %d unanswered",
            len(explanations),
            domain.name,
            language_model.name,
            cached,
            asked,
            unanswered,
        )
        return tuple(explanations)

    def _variables(self, domain: Domain) -> str:
        """The domain's variables as the initial state holds them, one line per variable or indexed base."""
        vocabulary = self._expression_generator.vocabulary(domain, (domain.initial_state,))
        lines: list[str] = []
        for base, indices in vocabulary.indices_by_base.items():
            example = ", ".join(repr(index) for index in sorted(indices, key=repr)[0])
            values = ", ".join(repr(value) for value in vocabulary.values_by_base[base])
            lines.append(f"- `{base}[{example}]`, and at each of its {len(indices)} indices: {values}")
        for name, values in vocabulary.values_by_variable.items():
            if "(" not in name:
                lines.append(f"- `{name}`: {', '.join(repr(value) for value in values)}")
        return "\n".join(lines)
