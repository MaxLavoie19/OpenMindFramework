# Architecture

This is the design OMF is being reworked into. `doc/refactoring-todo.md` tracks which parts are built.
`doc/open-questions.md` lists the parts not yet confirmed, and the contradictions and unknowns still to be decided.

## Purpose and user

OMF is a framework for solving AI problems, such as playing any kind of game from its rules.

An OMF user is a programmer who imports OMF and defines a game in it with rules and constraints. The instructions can
be minimal. OMF then learns to play the game by:
- inference;
- self-play;
- studying games;
- training models;
- reading literature.

**The integrator runs the game itself.** OMF doesn't. It simulates the game to decide: the RBS, the CSP, the search
and any other model that helps represent the game's state at a given time and predict its future state given a set of
actions.

OMF ships without the libraries a problem needs. A problem lives in its own project, which installs OMF, imports any
library its rules need, and registers its games under the `openmind.domains` entry points. Chess lives in the
OpenMindChess project, with python-chess.

## Vocabulary

- **State**: named data models, such as a grid named `board` or a scalar named `turn` holding `"O"`.
- **Action**: what an agent does, with its parameters, such as `place(row=2, col=3)` or `move(directionDegree=90)`. An
  action can take time to perform and can have a cooldown.
- **Joint action**: the actions every agent performs at once. All agents always act at once; in a game played in
  turns, only one agent has legal actions.
- **Outcome**: the new state the agents are in after a joint action.
- **Outcome distribution**: each possible outcome with its probability, given by the predictor. In a deterministic
  game there is a single outcome.
- **Payoff**: what each agent gets from an outcome, as the game defines it.
- **Utility**: what an outcome is worth to one agent, over all of its goals, weighed by that agent's goal weights.
- **Goal**: something an agent wants, with a weight for its role and situation. An agent may pursue several at once.
- **Preference**: the weight an agent gives a goal.
- **Context**: a compartment of knowledge, and a level in the hierarchy: a game, a variant, a relaxation, an agent
  model, macromanagement or micromanagement. Rulesets, beliefs, heuristics, models and tasks belong to a context.
- **Ruleset**: a part of a context holding the rules for one purpose, such as the game's simulation, the board
  position heuristic, the move heuristic, or one known agent's play style. A rule belongs to the rulesets that list it,
  with a weight in each.
- **Policy**: a way of playing tuned to a subset of the actions, such as fleeing moves or attacking moves. Choosing a
  policy narrows what the optimizer has to look at, before it picks the action within it.
- **Node**: a state with what has been worked out about it — its extracted features, and, in a search, what exploring
  it found. A feature is extracted by the first model that asks for it and shared with every model after, like a
  memoized call. A search builds nodes for its tree, and anything else valuing a position builds one too.
- **Heuristic**: a fast estimate. OMF uses three kinds:
  - the **position value**: what a state is worth to each agent;
  - the **move value**: what an action is worth in a state;
  - the **policy value**: which policy is worth following here.
- **Task**: something OMF can spend time on, such as valuing a position, predicting an outcome, self-play or reviewing
  a game.
- **Model**: one way to perform a task: rules, a lookup table, a decision tree, an ensemble, a DNN, … A step is solved
  by a set of models, one per task it needs. Models are chosen by:
  - their measured accuracy;
  - their cost, in processing time;
  - their explainability, if necessary.
- **Budget**: the real time available to act.
- **Belief**: a claim held by some holder, with a degree of certainty based on the supporting evidence, and with its
  accuracy, precision and source.
- **Opinion**: a subjective judgement by its holder, such as "good", "bad", "cheap" or "expensive". It uses any
  qualifier, and it is its holder's direct experience. What an agent thinks another agent's opinion is, is a belief.
- **Source**: how a belief was reached: a mechanism with its parameters, such as an inference, a decoder or direct
  experience.
- **Direct experience**: raw data as received, kept word for word, such as a text or a clip. An observation is a direct
  experience. Data extracted from it are beliefs whose sources reference it.
- **Agent model**: what OMF knows of one agent (heuristics, beliefs, goals), from a generic player down to one instance.

## Best effort

The agent always does the best it can in the given circumstances. It uses the model that fits best. If none exists, it
lives with it, since there is nothing it can do about it. It may note that it needs a way to solve that kind of problem,
and add that as a task.

## OMF runs continuously, always doing the next best task

OMF never waits idle. At every moment it performs the task worth the most right now. The candidates are:
- ponder the game: possible inferences, heuristics, policies or any other important aspect;
- play itself;
- review its games;
- study literature: imported games, texts decoded into rules, and other models' output such as engine evaluations;
- train models;
- play a game.

A game a player starts comes first. OMF plays its moves or plans within that game's clock, and goes back to other tasks
when time allows.

**Tasks live in the knowledge base.** Each task is recorded with:
- its value, over several measures weighed together by preferences:
  - utility gained;
  - precision gained;
  - time saved;
- its expected time;
- the evidence behind them.

**Values drift.** A task's value is a belief, updated from what each run of the task actually brought. Interpreting
low-level games can be very useful at first and useless once OMF is proficient.

**Deferring.** An important task may need more time than is available, because of conflicting goals. It is kept
pending, not dropped. For example, analysing an interesting position isn't worth it during a bullet game, but it
becomes worth it a few minutes later, instead of starting a new game right away.

**Roles** change what is valuable:
- A player maximizes its utility.
- A coach analyses games and gives useful feedback, as it plays or between games.

## Hierarchy of contexts

OMF works on a hierarchy of contexts. Each level is a problem of its own, with:
- its own state, actions and rules;
- its own policies, heuristics and models;
- its own time scale.

**A level can be simpler than its parent.** Life is real time, and OMF processes high-level events in real time. Inside
a chess game, much of that complexity is eliminated: the state is the board, time is the clock, and play goes in
turns.

**A parent delegates to a child.** For example, a real-time strategy game is split into levels:
- **The parent** is high-level macromanagement, such as economy, expansion and army composition. It sets the child's
  goals and budget:
  - The budget is either carved out of the parent's own or run alongside it. The parent decides which.
  - The goal of a sub-task is provided by the parent, for example coaching rather than winning.
- **The child** is unit micromanagement, such as moving and targeting units. It pursues those goals within its own
  context, and reports its outcome back to the parent.

**Payoff and rhetorical gain live at different levels.**
- Inside a game's context, the game's payoff is what counts.
- At the parent level, the result becomes rhetorical gain: progress toward the distance the agent wants, on the
  questions it finds problematic. By winning at chess, the winner creates distance: he is a winner, not a loser.

**Contexts are compartmentalized.** Rules, beliefs, heuristics, models and tasks belong to their context. Chess and
checkers don't mix, and macromanagement doesn't mix with micromanagement.

**Knowledge crosses between contexts only on purpose:**
- A variant inherits from its game.
- A model can be tried in another context, and is kept there only if it proves itself.
- A generic principle, such as mobility, can be tested in several games, each with its own measured value.

Each level runs the same machinery: the agent loop, the time management policy, policies and the optimizer. The
next-best-task choice runs at the top, across contexts.

**State abstraction.** A level sees an abstraction of the state, not every detail. In a coding agent, the full state is
very large: the file tree, the diff, the libraries, the environment, … The high level may hold a file's path but not
its content, and fetch the specifics only when a policy needs them, such as reading a file to edit it. Abstractions
are made by models, like any other task: decoders, summaries, or relaxations that drop detail.

**Planning fits the level.** Search is one model of planning, not a requirement:
- In chess, MCTS fits well.
- A coding agent's high level plans with policies such as testing, debugging and refactoring. The state is too large
  and the outcomes too uncertain for a tree search to pay off. How it plans instead is still open.

The time management policy picks the planning model for each level, like any other model.

## The agent loop under a real-time budget

OMF is a real-time AI. Each step draws on one budget:

| Step | What happens |
|---|---|
| Perceive | decoders turn inputs into structured data |
| Process | inference and belief updates in the knowledge base |
| Plan | a policy, the optimizer and the search choose an action |
| Communicate | rhetoric plans messages, encoders word them |
| Act | encoders turn the chosen action into an output |

At a high level, an agent in a game chooses between:
- **acting**: performing one of its legal actions in the game. In chess, that is playing the best move found so far,
  offering a draw or abandoning, all of them actions the game declares;
- **planning**: running the simulation to explore viable options.

Planning is always available. Acting depends on the game's rules: when it's black's turn in chess, white has no legal
action, so it can only plan.

### The time management policy

Every step is solved by several models at once: predicting outcomes, valuing a move for each player, valuing a state
for each player, and so on. The **time management policy** picks the set of models that solves the step, not one model.
It chooses from:
- the time available;
- the models available;
- what each one trades: its precision against its processing time.

It works at every level:
- theory of mind: how many agent models and nested beliefs to consider;
- inferences: how much deduction or relaxation;
- the position value heuristic;
- the move value heuristic;
- the policy value heuristic;
- the predictor;
- binning;
- the number of SDMCTS nodes to explore;
- which task to perform next.

The policy is trainable. It learns which choices paid off under which time pressure. It starts from a simple bootstrap
policy, and precisions and processing times are measured, never assumed.

**Every reading is timed**, and from those timings OMF builds a model of each model's processing time: its performance
profile. The profiles are what a specialized build is generated from later, such as a competitive chess solver shipping
with the parts it needs built in.

## Bootstrap

1. Load the game's definition. In chess, it includes:
   - the pieces' movement patterns;
   - their constraints: knights can jump over pieces and bishops can't;
   - when the king can castle;
   - when a pawn can take en passant or promote.
2. Load the game's policies. If it has none, create a default policy.
   - The default policy is an unnamed policy whose sub-goal must be populated.
   - At worst, it plays randomly.
   - A chess agent with free time should deduce by itself that it will need to prepare a policy. It then analyses the
     game's rules, facts and so on to emit credible heuristics.
3. Infer the heuristics, assuming the user gave only minimal instructions. Two ways:
   - **Relaxed problems**: remove constraints and measure how far a win is. For example: how far am I from a win if I
     can teleport my pieces? If the other player can't play? Relaxations aren't a fixed list. They emerge from the
     constraint sets an agent builds, to generate heuristics or to simplify a problem it is studying.
   - **Guiding principles**: for example, of two positions, prefer the one with the most options.
4. As OMF plays, train better-fitted or faster models of each task:
   - a lookup table;
   - a decision tree;
   - an ensemble;
   - a DNN;
   - or any other technique.

Bootstrapping may itself need training. The **bootstrapper** is a model that learns which ways of bootstrapping pay
off:
- which relaxations;
- which guiding principles;
- which default policies;
- which starting models;
- in what order to do the early tasks.

It learns across games and over time, so later games start better.

## Defining a game

The programmer defines a game with rules in the knowledge base, in the simulation ruleset of the game's context:

| Rule | Says |
|---|---|
| Players and initial state | who plays, and where the game starts |
| Values and constraints | which parameter values an action may take, and which combinations are legal |
| Effects | what an action leads to, with its probability |
| Joint effects | what the agents' actions, taken at once, lead to together |
| Durations and cooldowns | how long an action takes to perform, and how long before it is available again |
| Who acts | the constraints determine each player's legal actions in a state, reading the game's variables such as `turn` and `phase`; a player without options that turn can only plan |
| End and payoffs | when the game is over and what each agent gets: an end state declares a payoff per player, which the predictor gives with the outcome |
| Policies | the game's policies, if the programmer gives any |

OMF knows nothing of turns, phases, priority, draws, abandoning, clocks or boards. They belong to the game the integrator
implements, as its own variables, actions and rules, and OMF doesn't enforce them. A draw offer, for instance, is just
one of the game's actions; OMF sees only the effect of performing it.

Rules are Python: every constraint, effect and heuristic is the source of a Python expression or script, compiled once
and run against states. Any Python is allowed, imports and libraries included.

**Frozen and open rules.**
- **Rules hardcoded by an application** that implements OMF are frozen as they are by default. OMF never edits them.
- **Rules OMF deduced** can be edited freely. A new observation can justify replacing them with a model that better
  explains the data.
- **Open rules.** A rule or a ruleset can be declared open for modification. OMF then revises it like a deduced rule.
  Both can also be frozen: an integrator might freeze the chess simulation ruleset so that OMF doesn't modify the
  game's rules.
- **Copies of frozen rules.** A heuristic's ruleset can be modified at any time. To change a frozen rule or a frozen
  ruleset, a heuristic modifies a copy, and the original stays as it is. A heuristic can also have an open ruleset of
  its own.
- **A frozen ruleset has only frozen rules.**
- **Conflicts with a frozen rule.** When an observation conflicts with a frozen rule, OMF doesn't revise the rule. It
  raises a warning for the developer through the debug and logs module.

  Example: the developer forgot to implement en passant, but the engine running the game implements it. OMF can't
  foresee an en passant capture, and then one is played.

A **variant** or a **relaxation** is another context. It inherits the game's rules, except the ones it drops, and adds
its own.

### Data structures

OMF provides data structure definitions with the methods and actions relevant to them, which a game definition imports.
The grid comes first, since it is common in games. It provides:
- cells, neighbours and adjacency;
- rows, columns, diagonals and lines;
- directions and rays until blocked;
- distances;
- regions and boxes;
- symmetries;
- placing, moving and removing pieces.

Constraints, effects and heuristics use these methods, and inference can read them to generate heuristics. Other
structures are added when a game needs one.

### Actions in time

- An action can take time to perform, such as a motor moving into position.
- An action can have a cooldown: after a jump, you must fall back down before jumping again.
- The CSP excludes actions still cooling down.
- The predictor gives outcomes over time, including actions still in progress.
- Time is seconds on a timeline, not a count of moves, so actions can last and overlap.

### Several kinds of play

OMF must play games of every kind:
- **Played in turns**, like chess: an agent can't move when it isn't its turn.
- **Simultaneous**, like rock paper scissors: agents decide at once.
- **Non-deterministic**: an action has several outcomes, each with its probability.
- **With hidden information**: agents see only part of the state.
- **Zero-sum or not**. Payoffs are per agent, so OMF supports non-adversarial play. In a zero-sum game, a minimizing
  opponent is one model of the opponent among others (see "Agent models").

OMF assumes all players play at the same time, all the time. In a game played in turns, a player has no options outside
their turn, because the game's own constraints say so; OMF knows nothing of turns.

That is the most generic simulation, and it costs more. Where performance drops too much, a specialized model can take
over, like any other model: in chess, a vanilla MCTS with the assumption that a single player acts per turn.

## Valid actions: the constraint satisfaction solver

The CSP finds the valid values of an action's parameters in a state, from the game's value and constraint rules:
- **Discrete actions:** it gives every legal move.
- **Continuous actions:** it gives the valid ranges, and the optimizer searches within them.

## The predictor

The predictor takes:
- the current state;
- the agents' actions.

It emits an outcome distribution. In a deterministic game like chess, a move in a given state always leads to the same
outcome.

The predictor is a task like any other. It has several models:
- the game's declared effects rules;
- a lookup table;
- a decision tree;
- a DNN;
- any other learned model, trained from observed transitions.

The time management policy picks among them.

## The rule-based system is a model, not a role

A heuristic can be anything: a rule-based system (RBS), a DNN, or a set of relaxed constraints, such as the distance as
the bird flies. An RBS remains an RBS, and what it is used for is unrelated to what it is. It can hold a game's rules,
a heuristic, a predictor, or anything else a task needs.

OMF's RBS is its main explainable model:
- it is good at explaining;
- first-order logic can populate it with initial reasonable rules.

**Rulesets.** A context holds several rulesets, one per purpose:
- one for the game's simulation;
- one for the board position heuristic;
- one for the move heuristic;
- and so on, such as one per known agent when OMF tries to learn that agent's play style.

A ruleset has an id, a name, its context, the task it models, tags, whether it is open, and the ids of the rules that
belong to it; more may be added as needed, such as its role. A rule can belong to several rulesets, since OMF may create
many similar rulesets, for example one per policy.

A ruleset belongs to a context, and a rule belongs to the rulesets that list it. A rule's weight belongs to the
relation between a ruleset and the rule, so it weighs differently in each ruleset. Every relation carries a weight, even
where it isn't used yet. Weights are linear for now: a rule whose value is x contributes a·x. Some RBS would do better
with non-linear weights, such as a·x^b; that is left for later.

A ruleset is one model of its task, so any other model might replace it: a DNN, a lookup table, a decision tree, … The
RBS doesn't always win, even on explainability. In chess's early game, a lookup table is more performant, more accurate
and more explainable than any ruleset.

A ruleset can also become a tool for rationalization: where another model decides, a ruleset attempts to explain that
model's decision.

## Heuristics read nodes

A heuristic is given a node, not a bare state: the state, the game it is in, and the features extracted from it. A
model that uses features either extracts them or fetches them from where they are stored, and when several models want
the same features they are extracted once and shared.

- A rule-based heuristic reads a board's tactics, mobility, win chances and whatever else its rules name.
- A network heuristic may read none of that, and the node costs it nothing.

**A win always carries the payoff value.** Whatever the heuristic, a finished position is worth what the game paid,
never what a model guesses.

## Policies and the optimizer

A policy is a way of playing tuned to a subset of the actions. Choosing one narrows what the optimizer has to look at.

The set of all possible actions is too large to weigh one by one, and much of it is nonsense. A soldier who can move
in any direction with `move(directionDegree)` has fleeing moves, attacking moves and moves that make no sense at all.
So the agent picks how it wants to play — flee, charge, kite — and then picks the action that serves that best.

**What a policy is made of.** A policy is a pair of heuristics specialised to it:
- a **position value** heuristic, valuing the states the policy is after;
- a **move value** heuristic, valuing the moves that serve it.

A fleeing policy values a state by how safe it is, and a move by how much distance it puts between the soldier and the
enemy. Both follow the policy's sub-goal, which is what makes them specialised: the same game, judged another way.

Each heuristic is a model like any other, so a policy's position value can be a ruleset in one game and a network in
the next (see "Models and the time management policy").

**Where a policy comes from.** It can be manufactured — an agent is told to learn freeze, fight and flight — or
learned by policy optimization.

**The optimizer is an alternative to expanding.** Expanding means listing the valid actions and sorting them, which
the CSP and the heuristics do. Optimizing means producing the one action that best serves the goal, without listing
anything: an artillery piece doesn't expand every ballistic solution, it calculates the one that hits the target.

For a policy:
1. The CSP determines the valid values for the action.
2. Either the search expands those values and rates them with the policy's move value heuristic, or an optimizer
   solves for the action that serves the policy's sub-goal best.

In a discrete game with only the default policy, before its sub-goal is populated, expanding gives the legal moves.

**Policies are how time is traded.** Picking at random among the valid values is itself a policy, the one that saves
time when there is none. With time to spare, an agent follows several complementary policies and weighs what each one
proposes. In chess it might begin with one and, as it learns, come to know when to play aggressively and when to play
safe.

**Generative optimizers.** When the action space is too large to list, such as the text of an NLP agent or the edits
of a coding agent, the optimizer does not list the possible actions. It proposes solutions instead:
- A coding agent might have policies for testing, debugging, reviewing, tracing, writing and editing.
- The selected policy generates the edit that best fits its needs.
- The CSP then checks that proposal against the constraints, rather than enumerating every valid value.

Optimizers are models of the same task, from cheap to costly. The time management policy picks among them, and picks
between optimizing and expanding:
- a random pick among the valid values: the cheapest, for when there is no time to think;
- equations computing the value that serves the sub-goal, such as a firing solution;
- a controller holding a variable on target, such as a PID;
- a constraint solver working the values out one at a time: an optimizer can be a special kind of CSP, and then what
  it gives is legal by construction;
- a model proposing a solution, such as the edit a coding agent makes.

What an optimizer gives is checked against the constraints, unless the optimizer is itself the solver: equations and
controllers know the goal, not the rules.

**Control systems are optimizers.** A system with several inputs and several outputs is optimized the way control
engineering optimizes one, and it fits without changing anything:
- An optimizer gives the **next best step from the current position**, one action, whatever it computes it with.
- **Model predictive control's plan** is the series of steps the optimizers give as the search explores: the horizon
  is how far the search goes, not something an optimizer returns.
- **Model reference adaptive control** adapts its parameters as it runs, and what it adapted is its own data, kept
  where the model record says, so it outlives the run.

**The policy picker** says which policies are worth expanding, and which aren't worth considering at all. Picking at
random is rarely worth expanding; when it is — when there is no time to think — nothing else is worth considering.

The **policy value heuristic** picks which policies are worth following. A policy takes the utility of the move it
decided on, and that utility trains the policy value heuristic.

## Utility

A move's utility is the value of each of its outcomes multiplied by that outcome's probability, summed.

In a continuous probability space, outcomes are binned, and each bin's value is multiplied by its likelihood. Take a
trip to the casino:
- winning 0–100 $ means it wasn't worth the trip;
- 100–200 $ is break-even;
- 200–1000 $ means it was worth it.

Each bin multiplied by its likelihood shows the trip isn't worth it.

**Binning is a task with its own models:**
- uniform probability bins, sufficient for most decisions;
- bins placed around a decision threshold, such as three bins around the casino's break-even range;
- an RBS;
- any other model.

**Preferences live in the knowledge base**, like everything else OMF produces, so that an agent can look back on its
own choices and say what it preferred.

**Several goals at once.** An agent may weigh several goals at once. A chess coach might play to barely win, while
prioritizing teachable moments. An outcome's value is its value on each goal, weighed by the agent's goal weights for
its role and situation. The weights are preferences, and they can be learned.

**Some values are qualitative.** In chess a value is a number, such as centipawns. In rhetoric it is a judgement, such
as "sounds selfish", gauged the way we gauge temperature: freezing, cold, lukewarm, warm, hot, burning. A domain says
which kind of value it uses.
- What each word means is learned, for one agent or in general.
- It depends on the context, the audience, and so on. The same outcome may be gauged differently for different
  agents.
- Gauging another agent's valuation is a guess. Finding out whether it was right is a negotiation. The guess may have:
  - been right;
  - neglected a factor;
  - over-valued an existing factor;
  - under-valued an existing factor.

**Words are fuzzy sets over an underlying scale.**
- Each word has a membership curve on a scale, learned per agent or context.
- The search and utility compute on the scale, with fuzzy arithmetic: backups, averages, value × probability.
- The results are turned back into words when communicating or deciding.
- Fuzzy arithmetic is still arithmetic: one insult against two compliments can be computed, as a best-effort estimate.
- Values are estimated from incomplete information. They change from agent to agent, from time to time, and from
  context to context. Every value is a best effort.

**Soft goals.** Some goals are hard, like a checkmate or a passing test. Others are soft and fuzzy, like code quality
or user satisfaction:
- A soft goal has no exact measure. Its value is an opinion, held by the one who judges, often the user.
- Reaching a soft goal becomes a rhetorical problem. The agent must make a case, to the one who judges, that its work
  is at the best quality-for-effort ratio.
- An agent can edit its own opinions, which are its direct experience. It can't edit someone else's opinion.
- It can form beliefs about someone else's opinion, and update those beliefs, such as after hearing the judge.

## Search: semi-determinized Monte-Carlo tree search

SDMCTS is OMF's search, for the levels where a tree search fits (see "Planning fits the level"). It is guided by two
heuristics, as in AlphaZero:
- The **move value** heuristic is the prior of PUCT.
- The **position value** heuristic values the leaves.

There are no playouts to the end of the game.

**The simulation is how the search expands a node:**
1. the CSP provides the legal actions;
2. the heuristics choose between them;
3. the predictor provides the possible outcomes from the state and the set of actions.

- **Chance:** chance nodes branch on the predictor's outcome distribution.
- **Simultaneous play:** every node is a joint action of the agents with legal actions, and regret matching chooses
  where several agents act. Where one agent acts, as in chess, it reduces to that agent's choice.
- **Hidden information:** the search is semi-determinized. It runs over hypotheses about the hidden parts of the state,
  each weighted by the knowledge base's belief in it.
- **Other agents:** their replies come from their agent models' heuristics.
- **Budget:** the number of nodes to explore is set by the time management policy.

## Agent models, and theory of mind as an emergent property

Agents have different skill levels, knowledge and preferences, so OMF models specific agents when it can. The models
range from a generic player, through a class of players, down to one specific instance.

An agent model consists of:
- that agent's heuristics;
- that agent's goals;
- that agent's beliefs, held in the knowledge base with the agent as holder.

**Conflicting models are the norm.** The minimizer is only one model of an opponent:
- It suffers from projection: it plays the moves the agent would play itself.
- That might work against Stockfish, but it won't do for a low-level agent.

If another agent model better predicts an agent's actions and leads to better results, that model is preferred. As
with every model, the choice is based on:
- accuracy;
- cost;
- explainability, if necessary.

There is no theory-of-mind component. Theory of mind emerges from the knowledge base, the doxastic system and the agent
models' heuristics:
- what OMF believes another agent believes;
- what OMF believes that agent would value;
- how OMF predicts that agent would play.

## Detectors and explanations

Detectors recognize topics in a state, and serve as sources for them. In chess, detectors find patterns such as:
- a fork;
- a pin;
- a skewer;
- an x-ray.

A coach needs to be able to explain the board. Even when a move comes from an engine's evaluation, a self-made DNN or
another black-box model, OMF must be able to provide relevant information to rationalize the move. Detectors provide
that information.

**Detectors are decoders**, whose source is a state.

**Grounding is emergent** from decoders plus epistemology. There is no grounding module. The goal is to prevent
hallucination, such as an LLM producing a board analysis with no grounding in the move played or the board, sometimes
even making up pieces.

Chess tactics, such as a fork or a pin, are a different idiom from OMF's policies. The first belongs to chess; the second
is a general direction that guides the optimizer.

## Encoders and decoders

Several encoders and decoders can be connected to OMF:
- A **decoder** takes information from a source and converts it into structured data.
- An **encoder** takes structured data and turns it into an output.

Sources and outputs can be of any kind, such as:
- text;
- images;
- sound;
- temperature;
- radar;
- motor commands.

A decoder may give several interpretations of an input, each with its probability. That is where puns, ambiguity and
sarcasm come from.

## Rhetoric

Rhetoric is a built-in module that lets a player communicate with another. It follows Michel Meyer's rhetoric.

### How a message travels

1. A decoder turns a received message into structured data.
2. The player plans a reply.
3. An encoder turns the reply's structured data into natural language.
4. **Validation gate:** the encoder's output is first decoded, to ensure that the message at least sounds as intended.

### Meyer's model

- **Ethos** is the speaker. The effective ethos is who the speaker is; the projective ethos is who the speaker is
  perceived to be.
- **Pathos** is the audience. The effective pathos is who the audience member is; the projective pathos is who the
  speaker perceives them to be.
- **Positions** are answers to questions.
- A **distance** separates two positions on a question.
- **Problematicity** is how bad a distance is in itself:
  - A high problematicity means that the more distance there is, the more conflict there is.
  - A negative problematicity means that the more distance there is, the better. A teacher ought to have a much better
    understanding than their student; the more, the better.

### Rhetorical policies

Rhetoric's policies are policies like any other: each narrows the messages worth considering to those negotiating one
distance in one direction. They are permutations of:
- which distance: between the projective ethos, effective ethos, projective pathos and effective pathos;
- what the policy does to it: increase, affirm or reduce the distance, or its problematicity.

All 36 are pre-defined: 6 pairs of terms × 6 operations. Each learns to perform its specific job.

**A message carries several policies at once**, because more than one question is usually negotiated at a time. A
message must support them together: "If I could eat gold, I wouldn't starve; alas, during this siege even the king
can't buy food, so yes, I resorted to eating what I must to live another day" affirms nobility, affirms disgust at rat
meat, and reduces the problematicity of having eaten it, in one breath.

### Composing a message

Messages are prepared to achieve a balance between goals. An agent might want to stress the distance between their
soccer team and another agent's team, without stirring problems within their board game group.

A message can be composed of:
- questions;
- factoids: information, true or not, presented as fact;
- inferences;
- rules.

Its structured data can also contain stage directions: tone, vocabulary, specific phrasing or quotes, emphasis,
framing, … A decoder can recover some of them and not others.

A decoder may find several interpretations of a message, which gives rise to puns, ambiguity and sarcasm. Sarcasm is a
message that is clearly not intended as sincere, given the speaker's ethos.

Rhetoric always depends on epistemology. It is the only way to prevent hallucinations.

Rhetoric also serves soft goals (see "Utility"). An agent presenting its work, such as a feature, makes a case to its
judge that the work is at the best quality-for-effort ratio.

Distances are measured with fuzzy arithmetic (see "Utility"): best-effort estimates from incomplete information, which
vary by agent, time and context.

## Knowledge base and doxastic logic

Everything OMF produces is stored in the knowledge base, so that it can look back on it: what its models measured, what
it decided and why, as much as what it believes.

The knowledge base stores:
- rules and rulesets;
- models, each with a record of its own — its name, its task, its context and where its data lives — and its measured
  accuracy and processing time as beliefs;
- facts;
- beliefs;
- opinions;
- tasks;
- models.

A doxastic logic system tracks beliefs:
- the agent's own beliefs;
- the beliefs of other agents;
- nested beliefs: what the agent believes black believes white believes.

**Opinions are not beliefs.**
- An opinion is subjective: its holder's direct experience, with any qualifier, such as "good", "bad", "cheap" or
  "expensive".
- It has no certainty to weigh, because it isn't a claim about the world.
- What an agent thinks another agent's opinion is, is a belief, with its certainty and evidence.
- The knowledge base keeps opinions apart from beliefs.

A belief is not a label. The evidence for a claim and the evidence against it are kept apart, so an agent can hold poor
evidence for `p` and solid evidence for `not p`, and know that it does.

Every belief carries:
- **Certainty**: a degree of how strongly a belief is held. It should depend on evidence, but a belief might not have
  a known, proper epistemic justification.
- **Accuracy**: how often beliefs from the same source and method have turned out right.
- **Precision**: how narrow the belief is, such as the spread of an estimate.
- **Source**: the method the agent used to reach it:
  - **Direct experience**, such as the exact text as received: the source is the date, time, input and other
    relevant information.
  - **Interpreted, inferred, deduced or estimated**: the source is the method used and its parameters.

Everything is kept word for word, and nothing is overwritten.

**Storage.**
- OMF stores the knowledge base in JSON by default.
- The default is meant to be overridden by the integrator's long-term storage solution.
- The integrator is responsible for the data lifecycle, not OMF.

## Debug and logs

A module that helps a developer find problems in their game's definition and in OMF's reasoning. It owns logging and
warnings, and the dashboard is its viewer.

**Logging per session.**
- The log level depends on the session. Depending on what the user is doing, they might want minimal verbosity or
  targeted verbosity.
- This replaces a single global level.

**Debugger.** Given the complexity of the application, the module also provides what a debugger needs:
- interrupts;
- a stack, which is both, linked:
  - the Python call stack;
  - OMF's reasoning stack, such as task → policy → search node → evaluation → rule.

  Each reasoning frame points to the code it runs.
- time while paused, decided by the debug session: OMF's clocks either freeze or keep running;
- measurements taken across a pause, also decided by the debug session: they are either kept for training and timing
  statistics or left out;
- breakpoints, which can be conditional on the game. Example: break when you reach an evaluation for a board position
  that allows en passant.
- other debugger features.

**Warnings.** It raises warnings, such as:
- rules that conflict with each other;
- observations that a frozen rule can't explain, such as an en passant capture played when the definition lacks it.

## Epistemology: foundherentism

OMF is always learning, so it needs a theory of how its beliefs are justified. It follows foundherentism (Susan Haack),
which combines foundationalism and coherentism. Justification works like a crossword:
- the clues are the anchors: observations and fundamental rules;
- the crossing entries are the other beliefs that interlock with a belief.

**Anchors.** Anchors are what OMF has experienced directly, and the fundamental rules it was given, such as the game's
definition. They support beliefs without being derived from other beliefs. An anchor can still be wrong:
- A misperception, or a deduced rule or a rule declared open, is revised when the coherent whole is against it.
- A frozen rule is never revised. The conflict becomes a warning for the developer (see "Frozen and open rules").

**Coherence.** OMF strives for coherence between:
- its observations;
- its models;
- the fundamental rules.

When they disagree, it is a finding, not noise. The conflict becomes a warning and a task, since a model may have solved
a problem incorrectly and the proper inference may be something else: investigate it, test the model, or look for more
data.

**No self-supported claims.** A claim can't support itself. A circle of claims supporting each other, with no path back
to an anchor, gets no confidence from that circle. Support counts only as far as it traces back to anchors.

**Deduction and induction.** OMF uses the information it has to reason:
- **Deduction**: from rules and facts to what must follow. A deduction is as strong as its premises.
- **Induction**: from observations to general rules and models. An induction is as strong as its data and method allow.

**Confidence is nuanced by methodology and data.** A belief's confidence depends on how it was derived:
- the method, and its measured accuracy;
- how much data it used, and how representative that data was;
- how independent its supports are from one another.

For example:
- A rule proved from the game's definition is near certain.
- A heuristic fitted on a hundred games of self-play is tentative.
- A claim heard once from an unreliable teller is weak.

Every belief's source names its method and parameters, so its confidence can be re-evaluated when the method's
accuracy is re-measured.

## Frozen for production

OMF has a no-learning entrypoint, for when a trained model must be frozen and shipped in production. By default, it
freezes:
- trained models;
- rules: no new ones;
- policies: no new ones;
- facts: no new ones;
- beliefs: none are persisted.

Beliefs a task needs are kept in that task's instance context, and go away with it.

## Packages

This map is provisional: each package's boundaries are confirmed at its own interfaces stop. The packages are listed
in dependency order.

| Package | Owns |
|---|---|
| `structure` | the data models a state is made of, with OMF's methods: scalar, list, map, and grid in any number of dimensions |
| `world` | states made of named data models, actions, joint actions, players |
| `rule` | rules as data (Python source or a project's function), the shapes a game's rules take, and running a rule on a state |
| `knowledge` | the knowledge base, the doxastic system: direct experience (raw data, word for word), rules, rulesets, facts, beliefs with their sources, opinions, tasks, contexts, models |
| `epistemology` | justification of beliefs by foundherentism: anchors, coherence, no self-support, confidence from method and data |
| `game` | the programmer's API to define a game, the registry of games; the integrator runs the game, OMF only simulates it |
| `csp` | valid values of an action's parameters |
| `predictor` | the predictor port and its rule-based model |
| `rbs` | the rule-based system: a model family that runs whatever rules a ruleset holds, for any task; OMF's main explainable model |
| `model` | models per task, with their measured precision and processing time |
| `heuristic` | the position, move and policy value tasks: what any model filling them answers, whatever it is |
| `utility` | a move's utility over its outcomes, goals and their weights, binning |
| `policy` | policies, the default policy, the optimizer |
| `agent_model` | models of agents, from a generic player to one instance |
| `search` | semi-determinized MCTS |
| `budget` | time, clocks, deadlines, the time management policy |
| `codec` | encoders and decoders |
| `rhetoric` | Meyer's model, rhetorical policies, message composition, the validation gate |
| `inference` | relaxations, guiding principles, deduction, the bootstrapper |
| `training` | self-play, reviewing games, studying literature, training models |
| `agent` | the agent loop, one loop per level of the hierarchy with delegation between them, the continuous next-best-task loop, roles such as player and coach |
| `debug` | logging with per-session verbosity, warnings (conflicting rules, observations a frozen rule can't explain), a debugger (interrupts, stack, conditional breakpoints); the dashboard is its viewer |
| `dashboard` | a page to visualize and debug: tasks, time management choices, policies and the search tree, beliefs and sources, models, games |
| `entrypoint` | ways to run OMF: `openmind-play`, `openmind-solve`, `openmind-dashboard`, and a no-learning entrypoint that runs trained models frozen, for production |

`parallel` (worker processes) and `testing` (the pytest plugin that saves logs) are infrastructure that any package may
use.

## Conventions

- **Layout:**
  - Code lives in `src/openmind/<package>/<class type>/`.
  - Class types:
    - `model` (datatypes);
    - `factory` (functions that build an object with a builder and a recipe);
    - `builder`;
    - `service` (operations on data);
    - `repository` (storing and retrieving data);
    - `mapper` (format conversions);
    - `constant`;
    - `plugin` (pytest plugins).
  - Entrypoints live in `src/openmind/entrypoint/`.
  - Each package has a `README.md` covering its purpose, content, usage and logs.
- **Code:**
  - One class per module, named after it. A name that clashes with a Python keyword gets a trailing underscore
    (`not_.py`).
  - Models are frozen, slotted dataclasses, except the search tree's nodes, which change while searching.
  - Logic lives in services; I/O lives only in entrypoints.
  - Services are stateless, instantiated once when the application is built, and injected where they are needed.
    Models, such as an RBS, are data passed to the services that run them.
  - Nothing in OMF is specific to one game. Constants are learned or inferred, not hardcoded.
- **Logging:**
  - Services log through `logging.getLogger(__name__)`: what was decided and why, with the key values.
  - Actions appear in logs as readable text, and rules as their source.
  - Per-call details log at DEBUG; decisions log at INFO.
- **Tests:**
  - Unit tests sit beside their target as `<module>_tests.py`; integration and end-to-end tests live in `test/`.
  - Tests pin the desired behaviour.
  - Tests save their logs in `data/log/<test file>/<test name>.log`.
- **Documentation:** OMF needs many of them:
  - `README.md` files, written with each component;
  - instructions for coding agents that use OMF to write games and applications;
  - tutorials and examples, in a step of their own once enough works end to end.
- **Data:** `data/` holds all data (databases, trained models, logs, …) and git ignores it.
