export const meta = {
  name: 'shallow-research',
  description: 'Small-fan-out literature pass: scope into angles, search each, refute what they rest on, synthesize',
  whenToUse:
    'A cited answer from the literature without the cost of a full deep-research run. Fourteen agents by default rather than a hundred: one scoping, four searching, eight refuting, one synthesizing. Pass the question as args, or {question, angles, votes, claims} to widen or narrow it.',
  phases: [
    { title: 'Scope', detail: 'one agent turns the question into distinct angles' },
    { title: 'Search', detail: 'one agent per angle, reading primary sources' },
    { title: 'Check', detail: 'each angle’s load-bearing claims put to a refuter' },
    { title: 'Synthesize', detail: 'what survived, what did not, and what it means' },
  ],
}

// The question, and how wide to go. A bare string is the question at the default width.
const asked = typeof args === 'string' ? { question: args } : args || {}
const QUESTION = String(asked.question || '').trim()
// Four angles, two claims apiece and one refuter per claim is fourteen agents with the scope and the
// synthesis. They are here to be raised for a question worth more, not as a ceiling to work around, and
// `claims: 1` brings it down to ten.
const ANGLES = Math.max(1, Math.min(8, Number(asked.angles) || 4))
const VOTES = Math.max(0, Math.min(3, asked.votes === undefined ? 1 : Number(asked.votes)))
const CLAIMS = Math.max(1, Math.min(4, Number(asked.claims) || 2))

if (!QUESTION) {
  throw new Error('shallow-research needs a question: pass it as args, or {question, angles, votes, claims}')
}

const SCOPE = {
  type: 'object',
  properties: {
    angles: {
      type: 'array',
      description: 'The distinct angles to search, each one a self-contained brief for one agent',
      items: {
        type: 'object',
        properties: {
          key: { type: 'string', description: 'A short slug naming the angle' },
          brief: { type: 'string', description: 'What that agent is to find, name and quote, in full sentences' },
        },
        required: ['key', 'brief'],
      },
    },
  },
  required: ['angles'],
}

const REPORT = {
  type: 'object',
  properties: {
    report: { type: 'string', description: 'A dense factual report with citations, no padding' },
    claims: {
      type: 'array',
      description: 'The load-bearing claims from this angle, each stated so that it could be shown false',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          source: { type: 'string', description: 'The primary source, with a URL where there is one' },
        },
        required: ['claim', 'source'],
      },
    },
  },
  required: ['report', 'claims'],
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean', description: 'true where the claim is wrong, overstated, or unsupported by its source' },
    evidence: { type: 'string', description: 'What was read and what it said, quoting where it matters' },
  },
  required: ['refuted', 'evidence'],
}

phase('Scope')
const scoped = await agent(
  `Turn this question into exactly ${ANGLES} distinct research angles, each a brief for one agent that will search and read primary sources.

THE QUESTION:
${QUESTION}

Rules for the angles. They must not overlap: two agents finding the same paper is a wasted agent. At least one angle must be adversarial — the case AGAINST, the negative results, what would make this fail — because a search that only looks for support finds only support. Name actual authors, systems and papers wherever you can, so the agent starts from a lead rather than from a keyword. Say in each brief that primary sources are to be read rather than summaries of them, and that load-bearing statements are to be quoted.

Return ${ANGLES} angles.`,
  { label: 'scope', phase: 'Scope', schema: SCOPE },
)

const angles = (scoped && scoped.angles ? scoped.angles : []).slice(0, ANGLES)
if (!angles.length) {
  return { question: QUESTION, error: 'the scoping agent returned no angles' }
}
log(`${angles.length} angles: ${angles.map((one) => one.key).join(', ')}`)

const found = await pipeline(
  angles,
  (angle) =>
    agent(
      `${angle.brief}

Read primary sources rather than summaries of them: fetch the paper, and where a PDF is saved locally, open it. Quote anything load-bearing. Give concrete figures from the tables rather than the abstract's adjectives. Where you could not verify something, say so rather than repeating it. Return a dense factual report and do not pad it.`,
      { label: `search:${angle.key}`, phase: 'Search', schema: REPORT },
    ),
  (result, angle) => {
    if (!result) return null
    const claims = (result.claims || []).slice(0, CLAIMS)
    if (!VOTES || !claims.length) {
      return { angle: angle.key, report: result.report, claims: claims.map((one) => ({ ...one, refuted: null, why: 'not checked' })) }
    }
    const checks = []
    for (let at = 0; at < claims.length; at += 1) {
      for (let vote = 0; vote < VOTES; vote += 1) {
        checks.push(() =>
          agent(
            `Try to REFUTE this claim about the research literature.

CLAIM: ${claims[at].claim}
SOURCE GIVEN: ${claims[at].source}

Fetch the primary source and check it yourself; do not trust anybody's summary, including the one that produced this claim. Refute it where the source does not say it, where it says less than the claim does, where the claim generalises past what was measured, or where later work overturned it. Default to refuted=true where you cannot verify it against a primary source — an unverified claim is not a finding. Say what you actually read.`,
            { label: `check:${angle.key}:${at}:${vote}`, phase: 'Check', schema: VERDICT },
          ),
        )
      }
    }
    return parallel(checks).then((votes) => ({
      angle: angle.key,
      report: result.report,
      claims: claims.map((one, at) => {
        const mine = votes.slice(at * VOTES, (at + 1) * VOTES).filter(Boolean)
        const against = mine.filter((held) => held.refuted).length
        return {
          ...one,
          refuted: mine.length ? against * 2 > mine.length : null,
          votes: `${against} of ${mine.length} refuted`,
          why: mine.map((held) => held.evidence).join('\n\n') || 'not checked',
        }
      }),
    }))
  },
)

const kept = found.filter(Boolean)
const claims = kept.flatMap((one) => one.claims)
const killed = claims.filter((one) => one.refuted === true).length
log(`${kept.length} of ${angles.length} angles came back; ${killed} of ${claims.length} claims refuted`)

phase('Synthesize')
const synthesis = await agent(
  `Synthesize an answer to this question from the angle reports below.

THE QUESTION:
${QUESTION}

Write for an expert reader who wants to know what is established, what is contested, and what to do about it. Structure it:

1. WHAT IS ESTABLISHED — what survived the refutation checks, with figures and citations.
2. WHAT DID NOT SURVIVE — name the refuted claims plainly and say why they fell. Do not quietly drop them; a reader who was told something last week deserves to hear it was wrong.
3. WHAT IS CONTESTED — where the angles disagree, say so rather than averaging them.
4. THE LEADS WORTH FOLLOWING — ranked, one sentence each on why.
5. WHAT NOBODY HAS DONE — where the asker would be generating evidence rather than consuming it.

Cite. Give numbers. Do not pad, and do not restate the question back.

THE ANGLE REPORTS:
${JSON.stringify(kept, null, 1).slice(0, 120000)}`,
  { label: 'synthesize', phase: 'Synthesize' },
)

return {
  question: QUESTION,
  synthesis,
  refuted: claims.filter((one) => one.refuted === true).map((one) => ({ claim: one.claim, votes: one.votes, why: one.why })),
  standing: claims.filter((one) => one.refuted !== true).map((one) => ({ claim: one.claim, source: one.source, votes: one.votes })),
}
