import logging
from dataclasses import replace

from openmind.epistemology.service.accuracy_scorer import AccuracyScorer
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.constant.model_constant import PROCESSING_TIME, READINGS, RULES
from openmind.model.model.model_measure import ModelMeasure

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Which models perform which task, and how each one has been measured.

    A model is a record in the knowledge base; what it is measured at — its accuracy, its spread, the seconds a reading
    takes — are beliefs about it, so the epistemology weighs them like anything else the agent believes. This service
    keeps nothing itself: the knowledge base is given with every call.

    Until the time management policy arrives (the budget step), whatever reads a task takes the best measured model of
    it."""

    def __init__(self, accuracy_scorer: AccuracyScorer) -> None:
        self._accuracy_scorer = accuracy_scorer

    def register(self, knowledge_base: KnowledgeBase, model: ModelRecord) -> ModelRecord:
        """Keeps the model, or updates the one of the same name in the same context. Its mechanism is registered where
        it isn't yet, so its readings can be sourced by it and scored."""
        standing = knowledge_base.model_named(model.context, model.name)
        kept = knowledge_base.model(replace(model, id=model.id or ("" if standing is None else standing.id)))
        logger.info(
            "Registered %s as a model of %s in %s",
            knowledge_base.readable_model(kept.id),
            kept.task,
            knowledge_base.readable_context(kept.context),
        )
        return kept

    def register_ruleset(self, knowledge_base: KnowledgeBase, ruleset: Ruleset, name: str | None = None) -> ModelRecord:
        """Registers a ruleset as a model of its task: a rule-based system, sourced by the mechanism that produced the
        ruleset, and found again by the ruleset's id. Named after the ruleset unless another name is given."""
        return self.register(
            knowledge_base,
            ModelRecord(
                name or ruleset.name,
                ruleset.task,
                ruleset.context,
                RULES,
                ruleset.source.mechanism,
                ruleset.id,
            ),
        )

    def of_task(self, knowledge_base: KnowledgeBase, context_id: str, task: str) -> tuple[ModelRecord, ...]:
        """Every model of that task in the context, or, where it has none, in the contexts it inherits from, nearest
        first."""
        seen: set[str] = set()
        pending = [context_id]
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            found = knowledge_base.models(current, task)
            if found:
                return found
            context = knowledge_base.context_by_id(current)
            if context is not None:
                pending.extend(context.inherits)
        return ()

    def best(self, knowledge_base: KnowledgeBase, context_id: str, task: str) -> ModelRecord | None:
        """The task's model measured most accurate, ties going to the fastest, then to the one registered first; None
        where the task has no model. A model not measured yet counts as accurate as an unscored mechanism."""
        found = self.of_task(knowledge_base, context_id, task)
        if not found:
            return None
        def measured(model: ModelRecord) -> tuple[float, float]:
            measure = self.measured(knowledge_base, model)
            accuracy = -1.0 if measure.accuracy is None else measure.accuracy
            seconds = 0.0 if measure.processing_seconds is None else measure.processing_seconds
            return (-accuracy, seconds)
        chosen = min(found, key=measured)
        logger.debug(
            "The best model of %s in %s is %s",
            task,
            knowledge_base.readable_context(context_id),
            knowledge_base.readable_model(chosen.id),
        )
        return chosen

    def measured(self, knowledge_base: KnowledgeBase, model: ModelRecord) -> ModelMeasure:
        """What the model has been measured at in its own context."""
        timing = knowledge_base.belief(PROCESSING_TIME.format(model=model.id), model.context)
        return ModelMeasure(
            self._accuracy_scorer.accuracy(knowledge_base, model.mechanism, model.context),
            self._accuracy_scorer.spread(knowledge_base, model.mechanism, model.context),
            None if timing is None else float(timing.value),  # type: ignore[arg-type]
            0 if timing is None else int(dict(timing.tags).get(READINGS, 0)),  # type: ignore[arg-type]
        )
