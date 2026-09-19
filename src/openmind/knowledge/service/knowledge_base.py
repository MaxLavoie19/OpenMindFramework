import logging
from dataclasses import replace
from datetime import datetime

from openmind.knowledge.constant.knowledge_constant import (
    BELIEF,
    CONTEXT,
    COPY,
    DECLARATION,
    EXPERIENCE,
    MECHANISM,
    MODEL,
    OPINION,
    RULE,
    RULESET,
    TASK,
)
from openmind.knowledge.mapper.knowledge_json_mapper import KnowledgeJsonMapper
from openmind.knowledge.mapper.model_record_json_mapper import ModelRecordJsonMapper
from openmind.knowledge.mapper.rule_record_json_mapper import RuleRecordJsonMapper
from openmind.knowledge.mapper.ruleset_json_mapper import RulesetJsonMapper
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.context import Context
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.identifier import new_identifier
from openmind.knowledge.model.mechanism import Mechanism
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.model.opinion import Opinion
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.ruleset_link import RulesetLink
from openmind.knowledge.model.source import Source
from openmind.knowledge.model.store import Store
from openmind.knowledge.model.tags import Tags, carries
from openmind.knowledge.model.task import Task

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Everything the agent knows about a domain: its direct experiences, kept word for word; its beliefs, each a
    variable with a value, a certainty and optional evidence, including what it believes others believe; its own
    opinions; its tasks; its contexts; the mechanisms its evidence comes from; the rules it knows, the rulesets
    listing them, and the models that perform its tasks. Everything
    carries tags it can be retrieved by, and a GUID every link uses; contexts and mechanisms also have names, which
    people and applications use and which the knowledge base resolves to their ids.

    Each kind is kept in a store of its own, JSON lines by default and the integrator's storage otherwise, and held in
    memory once loaded. Nothing kept is rewritten in place: a belief, an opinion or a task updated is written anew under
    its id, and the store keeps every version."""

    def __init__(
        self,
        domain: str,
        experiences: Store,
        beliefs: Store,
        opinions: Store,
        tasks: Store,
        contexts: Store,
        mechanisms: Store,
        rules: Store,
        rulesets: Store,
        models: Store,
        knowledge_json_mapper: KnowledgeJsonMapper | None = None,
        rule_record_json_mapper: RuleRecordJsonMapper | None = None,
        ruleset_json_mapper: RulesetJsonMapper | None = None,
        model_record_json_mapper: ModelRecordJsonMapper | None = None,
    ) -> None:
        self._domain = domain
        self._experience_store = experiences
        self._belief_store = beliefs
        self._opinion_store = opinions
        self._task_store = tasks
        self._context_store = contexts
        self._mechanism_store = mechanisms
        self._rule_store = rules
        self._ruleset_store = rulesets
        self._model_store = models
        self._mapper = KnowledgeJsonMapper() if knowledge_json_mapper is None else knowledge_json_mapper
        self._rule_mapper = RuleRecordJsonMapper(self._mapper) if rule_record_json_mapper is None else rule_record_json_mapper
        self._ruleset_mapper = RulesetJsonMapper(self._mapper) if ruleset_json_mapper is None else ruleset_json_mapper
        self._model_mapper = ModelRecordJsonMapper(self._mapper) if model_record_json_mapper is None else model_record_json_mapper
        self._experiences: dict[str, DirectExperience] = {}
        self._beliefs: dict[str, Belief] = {}
        self._belief_ids: dict[tuple[str, str, tuple[str, ...]], str] = {}
        self._opinions: dict[str, Opinion] = {}
        self._opinion_ids: dict[tuple[str, str], str] = {}
        self._tasks: dict[str, Task] = {}
        self._contexts: dict[str, Context] = {}
        self._context_ids: dict[str, str] = {}
        self._mechanisms: dict[str, Mechanism] = {}
        self._mechanism_ids: dict[str, str] = {}
        self._rules: dict[str, RuleRecord] = {}
        self._rulesets: dict[str, Ruleset] = {}
        self._models: dict[str, ModelRecord] = {}
        self._load()

    @property
    def domain(self) -> str:
        return self._domain

    # direct experience

    def experience(self, experience: DirectExperience) -> DirectExperience:
        """Keeps the raw data word for word and gives it back with its id; the time it arrived is now when not given."""
        kept = replace(
            experience,
            id=experience.id or new_identifier(EXPERIENCE),
            at=experience.at if experience.at is not None else datetime.now(),
        )
        self._experiences[kept.id] = kept
        self._experience_store.append(self._mapper.experience_to_data(kept))
        logger.debug(
            "Experienced %s %s in %s from %s",
            kept.id,
            kept.variable,
            self.readable_context(kept.context),
            self.readable_mechanism(kept.source.mechanism),
        )
        return kept

    def experienced(self, experience_id: str) -> DirectExperience | None:
        return self._experiences.get(experience_id)

    def experiences(self, context: str | None = None, tags: Tags = ()) -> tuple[DirectExperience, ...]:
        """Every direct experience in that context (an id) carrying all the tags, in the order they came in."""
        return tuple(
            experience
            for experience in self._experiences.values()
            if (context is None or experience.context == context) and carries(experience.tags, tags)
        )

    # beliefs

    def believe(self, belief: Belief) -> Belief:
        """Sets the belief as given, or updates the one held for the same variable, context and holder, which keeps its
        id; gives it back with its id."""
        belief_id = self._belief_ids.get(belief.key) or belief.id or new_identifier(BELIEF)
        kept = replace(belief, id=belief_id)
        self._beliefs[belief_id] = kept
        self._belief_ids[kept.key] = belief_id
        self._belief_store.append(self._mapper.belief_to_data(kept))
        logger.debug(
            "Believes %s = %r in %s%s at %.3g (%s)",
            kept.variable,
            kept.value,
            self.readable_context(kept.context),
            f" as held by {' > '.join(kept.holder)}" if kept.holder else "",
            kept.certainty,
            kept.id,
        )
        return kept

    def belief(self, variable: str, context: str, holder: tuple[str, ...] = ()) -> Belief | None:
        belief_id = self._belief_ids.get((variable, context, holder))
        return None if belief_id is None else self._beliefs[belief_id]

    def belief_by_id(self, belief_id: str) -> Belief | None:
        return self._beliefs.get(belief_id)

    def beliefs(
        self, context: str | None = None, holder: tuple[str, ...] | None = None, tags: Tags = ()
    ) -> tuple[Belief, ...]:
        """Every belief in that context (an id), held by that holder, carrying all the tags, in the order first
        believed."""
        return tuple(
            belief
            for belief in self._beliefs.values()
            if (context is None or belief.context == context)
            and (holder is None or belief.holder == holder)
            and carries(belief.tags, tags)
        )

    # opinions

    def hold(self, opinion: Opinion) -> Opinion:
        """Sets the agent's own opinion, or updates the one it holds on the same variable in the same context; gives it
        back with its id."""
        key = (opinion.variable, opinion.context)
        opinion_id = self._opinion_ids.get(key) or opinion.id or new_identifier(OPINION)
        kept = replace(opinion, id=opinion_id, at=opinion.at if opinion.at is not None else datetime.now())
        self._opinions[opinion_id] = kept
        self._opinion_ids[key] = opinion_id
        self._opinion_store.append(self._mapper.opinion_to_data(kept))
        logger.debug("Holds %s %r in %s (%s)", kept.variable, kept.value, self.readable_context(kept.context), kept.id)
        return kept

    def opinion(self, variable: str, context: str) -> Opinion | None:
        opinion_id = self._opinion_ids.get((variable, context))
        return None if opinion_id is None else self._opinions[opinion_id]

    def opinions(self, context: str | None = None, tags: Tags = ()) -> tuple[Opinion, ...]:
        return tuple(
            opinion
            for opinion in self._opinions.values()
            if (context is None or opinion.context == context) and carries(opinion.tags, tags)
        )

    # tasks

    def task(self, task: Task) -> Task:
        """Adds the task, or updates it under its id; gives it back with its id."""
        kept = replace(task, id=task.id or new_identifier(TASK))
        self._tasks[kept.id] = kept
        self._task_store.append(self._mapper.task_to_data(kept))
        logger.debug("Task %s in %s is %s (%s)", kept.name, self.readable_context(kept.context), kept.status, kept.id)
        return kept

    def tasks(self, context: str | None = None, status: str | None = None, tags: Tags = ()) -> tuple[Task, ...]:
        return tuple(
            task
            for task in self._tasks.values()
            if (context is None or task.context == context)
            and (status is None or task.status == status)
            and carries(task.tags, tags)
        )

    # contexts

    def context(self, context: Context) -> Context:
        """Registers the context, or updates the one with the same id."""
        self._contexts[context.id] = context
        self._context_ids[context.name] = context.id
        self._context_store.append(self._mapper.context_to_data(context))
        logger.debug(
            "Context %s%s%s",
            self.readable_context(context.id),
            f" in {self.readable_context(context.parent)}" if context.parent else "",
            f", inheriting from {', '.join(self.readable_context(each) for each in context.inherits)}" if context.inherits else "",
        )
        return context

    def ensure_context(self, name: str, parent: str | None = None) -> Context:
        """The context of that name, registered with a new id the first time it is named."""
        known = self.context_named(name)
        return known if known is not None else self.context(Context(new_identifier(CONTEXT), name, parent))

    def context_named(self, name: str) -> Context | None:
        context_id = self._context_ids.get(name)
        return None if context_id is None else self._contexts[context_id]

    def context_by_id(self, context_id: str) -> Context | None:
        return self._contexts.get(context_id)

    def contexts(self, tags: Tags = ()) -> tuple[Context, ...]:
        """Every context registered, carrying all the tags, in the order first registered."""
        return tuple(context for context in self._contexts.values() if carries(context.tags, tags))

    def readable_context(self, context_id: str | None) -> str:
        """A context as logs show it: its name and its id."""
        context = None if context_id is None else self._contexts.get(context_id)
        return f"{context.name} ({context.id})" if context is not None else str(context_id)

    # mechanisms

    def mechanism(self, mechanism: Mechanism) -> Mechanism:
        """Registers the mechanism, or updates the one with the same id."""
        self._mechanisms[mechanism.id] = mechanism
        self._mechanism_ids[mechanism.name] = mechanism.id
        self._mechanism_store.append(self._mapper.mechanism_to_data(mechanism))
        logger.debug("Mechanism %s", self.readable_mechanism(mechanism.id))
        return mechanism

    def ensure_mechanism(self, name: str, declared_accuracy: float | None = None) -> Mechanism:
        """The mechanism of that name, registered with a new id the first time it is named."""
        known = self.mechanism_named(name)
        return known if known is not None else self.mechanism(Mechanism(new_identifier(MECHANISM), name, declared_accuracy))

    def mechanism_named(self, name: str) -> Mechanism | None:
        mechanism_id = self._mechanism_ids.get(name)
        return None if mechanism_id is None else self._mechanisms[mechanism_id]

    def mechanism_by_id(self, mechanism_id: str) -> Mechanism | None:
        return self._mechanisms.get(mechanism_id)

    def mechanisms(self, tags: Tags = ()) -> tuple[Mechanism, ...]:
        return tuple(mechanism for mechanism in self._mechanisms.values() if carries(mechanism.tags, tags))

    def readable_mechanism(self, mechanism_id: str) -> str:
        """A mechanism as logs show it: its name and its id."""
        mechanism = self._mechanisms.get(mechanism_id)
        return f"{mechanism.name} ({mechanism.id})" if mechanism is not None else mechanism_id

    # rules

    def declare(self, rule: RuleRecord) -> RuleRecord:
        """Keeps the rule and gives it back with the id it can be found by. A rule already carrying an id is written
        anew under that id. This is how an application registers the rules of its game and how inference injects the
        ones it produces; a ruleset then lists it (see `link`)."""
        kept = rule
        if not kept.id:
            kept = replace(kept, id=new_identifier(RULE))
        if kept.source.at is None:
            kept = replace(kept, source=replace(kept.source, at=datetime.now()))
        self._rule_store.append(self._rule_mapper.to_data(kept))
        self._rules[kept.id] = kept
        logger.debug("Declared %s rule %s (%s)", kept.kind, kept.name, kept.id)
        return kept

    def revise(self, rule_id: str, rule: RuleRecord) -> RuleRecord:
        """Replaces the rule under that id with the revision, unless the rule is frozen: an application declared it and
        didn't declare it open. A frozen rule is left as it is, and the refusal is logged as a warning for the
        developer."""
        held = self._rules.get(rule_id)
        if held is None:
            raise KeyError(f"No rule {rule_id} to revise")
        if self.frozen(held):
            logger.warning(
                "Rule %s (%s) is frozen: an application declared it and didn't declare it open; the revision is refused",
                held.name,
                held.id,
            )
            return held
        return self.declare(replace(rule, id=rule_id))

    def frozen(self, rule: RuleRecord) -> bool:
        """Whether OMF must leave the rule as it is: declared by an application, and not declared open."""
        declaration = self.mechanism_named(DECLARATION)
        return declaration is not None and rule.source.mechanism == declaration.id and not rule.open

    def rule(self, rule_id: str) -> RuleRecord | None:
        return self._rules.get(rule_id)

    def undeclare(self, rule_id: str) -> None:
        """Drops the rule: it is retrieved no more."""
        if self._rules.pop(rule_id, None) is None:
            return
        self._rule_store.forget(rule_id)

    # rulesets

    def ruleset(self, ruleset: Ruleset) -> Ruleset:
        """Keeps the ruleset, or writes it anew under its id, and gives it back with its id. A frozen ruleset lists only
        frozen rules: one listing an open rule is refused with a warning, and what was kept before stays."""
        kept = replace(ruleset, id=ruleset.id or new_identifier(RULESET))
        if kept.source.at is None:
            kept = replace(kept, source=replace(kept.source, at=datetime.now()))
        if self.frozen_ruleset(kept):
            unfrozen = [rule_id for rule_id in kept.rule_ids if (rule := self._rules.get(rule_id)) is not None and not self.frozen(rule)]
            if unfrozen:
                logger.warning(
                    "Ruleset %s is frozen and lists only frozen rules; %s %s open, so the change is refused",
                    self.readable_ruleset(kept.id) if kept.id in self._rulesets else f"{kept.name} ({kept.id})",
                    ", ".join(self._readable_rule(rule_id) for rule_id in unfrozen),
                    "is" if len(unfrozen) == 1 else "are",
                )
                held = self._rulesets.get(kept.id)
                return held if held is not None else replace(kept, links=tuple(link for link in kept.links if link.rule not in unfrozen))
        self._rulesets[kept.id] = kept
        self._ruleset_store.append(self._ruleset_mapper.to_data(kept))
        logger.debug(
            "Ruleset %s of %s in %s lists %d rules%s",
            self.readable_ruleset(kept.id),
            kept.task,
            self.readable_context(kept.context),
            len(kept.links),
            "" if self.frozen_ruleset(kept) else ", open",
        )
        return kept

    def ruleset_by_id(self, ruleset_id: str) -> Ruleset | None:
        return self._rulesets.get(ruleset_id)

    def ruleset_named(self, context_id: str, name: str) -> Ruleset | None:
        for ruleset in self._rulesets.values():
            if ruleset.context == context_id and ruleset.name == name:
                return ruleset
        return None

    def rulesets(self, context_id: str | None = None, task: str | None = None, tags: Tags = ()) -> tuple[Ruleset, ...]:
        """Every ruleset of that context (an id), a model of that task, carrying all the tags, in the order first kept."""
        return tuple(
            ruleset
            for ruleset in self._rulesets.values()
            if (context_id is None or ruleset.context == context_id)
            and (task is None or ruleset.task == task)
            and carries(ruleset.tags, tags)
        )

    def ruleset_rules(
        self, ruleset_id: str, kinds: tuple[str, ...] = (), tags: Tags = ()
    ) -> tuple[tuple[RuleRecord, float], ...]:
        """The rules the ruleset lists, of those kinds or of any kind where none is named, carrying all the tags, each
        with its weight there, in the order they were linked."""
        ruleset = self._rulesets.get(ruleset_id)
        if ruleset is None:
            raise KeyError(f"No ruleset {ruleset_id}")
        found = []
        for link in ruleset.links:
            rule = self._rules.get(link.rule)
            if rule is not None and (not kinds or rule.kind in kinds) and carries(rule.tags, tags):
                found.append((rule, link.weight))
        return tuple(found)

    def link(self, ruleset_id: str, rule_id: str, weight: float = 1.0) -> Ruleset:
        """Lists the rule in the ruleset at that weight, or sets its weight there where it is listed already. An open
        rule isn't linked into a frozen ruleset: that is refused with a warning."""
        ruleset = self._rulesets.get(ruleset_id)
        if ruleset is None:
            raise KeyError(f"No ruleset {ruleset_id}")
        if rule_id not in self._rules:
            raise KeyError(f"No rule {rule_id} to link")
        if ruleset.weight(rule_id) is None:
            links = (*ruleset.links, RulesetLink(rule_id, weight))
        else:
            links = tuple(RulesetLink(rule_id, weight) if link.rule == rule_id else link for link in ruleset.links)
        return self.ruleset(replace(ruleset, links=links))

    def unlink(self, ruleset_id: str, rule_id: str) -> Ruleset:
        """Takes the rule off the ruleset; the rule itself stays, in every other ruleset listing it."""
        ruleset = self._rulesets.get(ruleset_id)
        if ruleset is None:
            raise KeyError(f"No ruleset {ruleset_id}")
        return self.ruleset(replace(ruleset, links=tuple(link for link in ruleset.links if link.rule != rule_id)))

    def frozen_ruleset(self, ruleset: Ruleset) -> bool:
        """Whether OMF must leave the ruleset as it is: declared by an application, and not declared open."""
        declaration = self.mechanism_named(DECLARATION)
        return declaration is not None and ruleset.source.mechanism == declaration.id and not ruleset.open

    def copy_ruleset(self, ruleset_id: str, name: str, context_id: str | None = None) -> Ruleset:
        """An open copy of the ruleset, listing the same rules at the same weights, in the same context unless another
        is given. Its source is the copy mechanism, resting on the original."""
        original = self._rulesets.get(ruleset_id)
        if original is None:
            raise KeyError(f"No ruleset {ruleset_id} to copy")
        copied = self.ruleset(
            Ruleset(
                name,
                original.context if context_id is None else context_id,
                original.task,
                Source(self.ensure_mechanism(COPY).id, (("original", original.id),), rests_on=(original.id,)),
                original.links,
                open=True,
                tags=original.tags,
            )
        )
        logger.info("Copied ruleset %s as %s", self.readable_ruleset(original.id), self.readable_ruleset(copied.id))
        return copied

    def revise_in(self, ruleset_id: str, rule_id: str, revision: RuleRecord) -> RuleRecord:
        """Revises a rule as one open ruleset lists it. An open rule is revised in place. A frozen rule stays as it is:
        the revision is kept as a copy, open and resting on the original, and it replaces the original in this ruleset
        only, at the same weight, so every other ruleset keeps the original. A frozen ruleset is left as it is, with a
        warning."""
        ruleset = self._rulesets.get(ruleset_id)
        if ruleset is None:
            raise KeyError(f"No ruleset {ruleset_id}")
        weight = ruleset.weight(rule_id)
        if weight is None:
            raise KeyError(f"Ruleset {ruleset_id} doesn't list rule {rule_id}")
        held = self._rules[rule_id]
        if self.frozen_ruleset(ruleset):
            logger.warning(
                "Ruleset %s is frozen: the revision of %s is refused; revise it in a copy",
                self.readable_ruleset(ruleset_id),
                self._readable_rule(rule_id),
            )
            return held
        if not self.frozen(held):
            return self.declare(replace(revision, id=rule_id))
        copy = self.declare(
            replace(
                revision,
                id="",
                open=True,
                source=replace(revision.source, rests_on=(*revision.source.rests_on, rule_id)),
            )
        )
        links = tuple(RulesetLink(copy.id, link.weight) if link.rule == rule_id else link for link in ruleset.links)
        self.ruleset(replace(ruleset, links=links))
        logger.info(
            "Rule %s is frozen: revised as its copy %s in ruleset %s only",
            self._readable_rule(rule_id),
            self._readable_rule(copy.id),
            self.readable_ruleset(ruleset_id),
        )
        return copy

    def readable_ruleset(self, ruleset_id: str) -> str:
        """A ruleset as logs show it: its name and its id."""
        ruleset = self._rulesets.get(ruleset_id)
        return f"{ruleset.name} ({ruleset.id})" if ruleset is not None else ruleset_id

    def _readable_rule(self, rule_id: str) -> str:
        rule = self._rules.get(rule_id)
        return f"{rule.name} ({rule.id})" if rule is not None else rule_id

    # models

    def model(self, model: ModelRecord) -> ModelRecord:
        """Keeps the model, or writes it anew under its id, and gives it back with its id."""
        kept = replace(model, id=model.id or new_identifier(MODEL))
        self._models[kept.id] = kept
        self._model_store.append(self._model_mapper.to_data(kept))
        logger.debug(
            "Model %s of %s in %s, a %s",
            self.readable_model(kept.id),
            kept.task,
            self.readable_context(kept.context),
            kept.family,
        )
        return kept

    def model_by_id(self, model_id: str) -> ModelRecord | None:
        return self._models.get(model_id)

    def model_named(self, context_id: str, name: str) -> ModelRecord | None:
        for model in self._models.values():
            if model.context == context_id and model.name == name:
                return model
        return None

    def models(self, context_id: str | None = None, task: str | None = None, tags: Tags = ()) -> tuple[ModelRecord, ...]:
        """Every model of that context (an id), performing that task, carrying all the tags, in the order first kept."""
        return tuple(
            model
            for model in self._models.values()
            if (context_id is None or model.context == context_id)
            and (task is None or model.task == task)
            and carries(model.tags, tags)
        )

    def readable_model(self, model_id: str) -> str:
        """A model as logs show it: its name and its id."""
        model = self._models.get(model_id)
        return f"{model.name} ({model.id})" if model is not None else model_id

    def _load(self) -> None:
        """Takes in what the stores already hold."""
        for data in self._context_store.load():
            context = self._mapper.context_from_data(data)
            self._contexts[context.id] = context
            self._context_ids[context.name] = context.id
        for data in self._mechanism_store.load():
            mechanism = self._mapper.mechanism_from_data(data)
            self._mechanisms[mechanism.id] = mechanism
            self._mechanism_ids[mechanism.name] = mechanism.id
        for data in self._experience_store.load():
            experience = self._mapper.experience_from_data(data)
            self._experiences[experience.id] = experience
        for data in self._belief_store.load():
            belief = self._mapper.belief_from_data(data)
            self._beliefs[belief.id] = belief
            self._belief_ids[belief.key] = belief.id
        for data in self._opinion_store.load():
            opinion = self._mapper.opinion_from_data(data)
            self._opinions[opinion.id] = opinion
            self._opinion_ids[(opinion.variable, opinion.context)] = opinion.id
        for data in self._task_store.load():
            task = self._mapper.task_from_data(data)
            self._tasks[task.id] = task
        for data in self._rule_store.load():
            rule = self._rule_mapper.from_data(data)
            self._rules[rule.id] = rule
        for data in self._ruleset_store.load():
            ruleset = self._ruleset_mapper.from_data(data)
            self._rulesets[ruleset.id] = ruleset
        for data in self._model_store.load():
            model = self._model_mapper.from_data(data)
            self._models[model.id] = model
        if self._experiences or self._beliefs or self._rules or self._contexts:
            logger.info(
                "Knowledge of %s: %d direct experiences, %d beliefs, %d opinions, %d tasks, %d contexts, %d mechanisms, "
                "%d rules, %d rulesets, %d models",
                self._domain,
                len(self._experiences),
                len(self._beliefs),
                len(self._opinions),
                len(self._tasks),
                len(self._contexts),
                len(self._mechanisms),
                len(self._rules),
                len(self._rulesets),
                len(self._models),
            )
