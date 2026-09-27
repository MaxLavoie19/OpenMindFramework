---
description: Small cited literature pass — scope, search, refute, synthesize
---

Run the `shallow-research` workflow on the question below.

It is the cheap counterpart to `/deep-research`: fourteen agents rather than a hundred, and minutes rather
than half an hour. One agent turns the question into four non-overlapping angles, one of which is always the case
*against*; four agents search and read primary sources; each angle's load-bearing claims go to a refuter that
defaults to refuted where it cannot check them; one agent synthesizes what survived and says plainly what did
not.

Before invoking, check the question is specific enough to search. Where it is not, ask one or two clarifying
questions and weave the answers in — a vague question spends fourteen agents on vagueness.

Invoke it as:

```
Workflow({ name: "shallow-research", args: "<the refined question>" })
```

Widen it only where the question earns it, by passing an object instead:

- `angles` — how many search agents, 1 to 8, four by default
- `votes` — refuters per claim, 0 to 3, one by default; three gives a majority verdict
- `claims` — load-bearing claims checked per angle, 1 to 4, two by default

The default is 1 + 4 + 8 + 1 = 14. `claims: 1` makes it ten; `{ angles: 6, votes: 3, claims: 2 }` is
1 + 6 + 36 + 1 = 44. Say what a widened run will cost before starting it.

When it returns, report the synthesis in your own words and **name the refuted claims out loud**. A claim that
did not survive is the most useful thing the run produces, and it is the first thing a summary drops.

The question:

$ARGUMENTS
