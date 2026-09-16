---
name: design-interview
description: Interviews the user (the researcher/owner) about a design before any code gets automated, so the implementation agent has zero unresolved assumptions. Use before plan-feature, before execute, and before launching any Archon mutation workflow, whenever a change is going to be automated. Produces a signed-off Decision Record the implementation MUST follow.
argument-hint: [optional feature description or Jira/Confluence refs]
---

# Design Interview: Reduce Assumptions to Zero

## Why This Skill Exists

A coding agent that omits a step risks skipping it. A coding agent that decides something for you decides it wrong more often than it decides it right, and it will be confident either way. This skill exists to make the **human the principal**: every decision that will shape the automated change is surfaced to you, asked, and recorded as a binding decision — not guessed.

Running order in this pack:

```
prime          -> load external context + map the codebase
design-interview -> GRILL THE USER, produce docs/decisions/<date>-<slug>.md   <- YOU ARE HERE
plan-feature   -> turn that decision record into a one-pass implementation plan
execute        -> implement the plan directly (interactive path)
  OR
archon researcher-piv -> implement the plan as a deterministic, rule-based workflow (Archon path)
```

The Decision Record you produce here is the **contract** between the interactive brain (this harness) and the rule-based engine (Archon). Do not hand anything to an automated implementer until this skill has completed with **zero open questions**.

## Input

Optional: a feature description, Jira/Confluence references, or a branch/TODO. If the session was primed, build on that context; otherwise do a light recap first:

- root `CLAUDE.md` / global rules
- root `README.md`
- the specific files the change will touch (skim, don't deep-dive — planning is a later step)

## Hard Rules

1. **Never guess.** Every assumption you hold about the change that the user did not author is material for a question.
2. **Use the `ask` tool** to interview. It is the equivalent of Claude Code's "ask user question" — do not ask free-form in chat and do not proceed on your own reading.
3. **At most ~5 questions per `ask` call.** Batch related questions into one call. Iterate: run a round, reconcile, run the next round, until no open questions remain.
4. **Each question must be concrete and offer tradeoffs.** 2–5 short option labels; set a recommended index using your codebase knowledge; mark `multi: true` only when several options genuinely apply. If the user picks "Other" or types free-form, capture their exact wording.
5. **"Use your judgement" is a decision, not an assumption.** When the user delegates a choice, record it in the **Delegated decisions** section with the user's delegation verbatim. It is then no longer an assumption — it is an authorized decision you own the consequences of. If you later think a delegated decision is wrong, pause and re-ask, do not silently change it.
6. **Do not move on while `## Open Questions` is non-empty.** Plan-feature and the Archon workflows are instructed to fail/skip if the decision record is missing or open.
7. **Never editorially revise a decision after implementation starts.** A later change of mind is a new decision: append a `## Revisions` entry noting the override and ask the user to confirm. The record is append-only.

## Mandatory Question Set

Cover every dimension below. You may skip a dimension only if the change provably cannot touch it (say so explicitly in the record).

| # | Dimension | What you are extracting |
|---|-----------|-------------------------|
| 1 | **Goal & value** | What changes, for whom, what counts as success in their terms. |
| 2 | **Scope boundaries** | What is explicitly IN, what is explicitly OUT. Absence of an explicit OUT costs you later — ask. |
| 3 | **User-facing behavior** | Exact behavior changes: inputs, outputs, states, transitions. |
| 4 | **Architecture & tech choices** | Where it plugs in, which existing patterns/components to reuse or extend, libraries to add or avoid, interface contracts. |
| 5 | **Data & contracts** | Schemas, persistence, API signatures, formats, migration concerns. |
| 6 | **Errors, edge cases, failure modes** | What should happen when things go wrong; acceptable degradation; non-goals for robustness. |
| 7 | **Validation & acceptance** | How you will know it is right: tests, evals, metrics, ablations, manual checks, browser verification. Be research-appropriate (reproducibility, baselines). |
| 8 | **Constraints & risks** | Performance, security, compliance, reproducibility, reversibility, and anything you judge risky. |
| 9 | **Hotspot: your own assumptions** | Every assumption you noticed in reading the codebase that the change depends on. |

## Process

### Round 1: Scope and shape
Ask dimensions 1–3 plus anything blocking a frame of reference. Goal: a crisp problem statement and boundaries.

### Round 2: Technical commitments
Ask dimensions 4–6. This is the "help me choose" round — present 2–4 alternatives per decision with tradeoffs, recommend one, let the user override.

### Round 3: Verification and risk
Ask dimensions 7–9, including any residual or newly surfaced assumptions. Also explicitly offer: *"Is there anything you want to defer to me that you haven't?"* — write answers verbatim into Delegated decisions.

### Reconcile and iterate
After each round, summarize what you believe the change to be *in one short paragraph* and state your outstanding assumptions explicitly. Loop further rounds only while genuine unknowns remain. **Prefer fewer, sharper rounds to more, fluffier ones.**

## Output: the Decision Record

Write to `docs/decisions/YYYY-MM-DD-<kebab-case-slug>.md` (create the directory if absent). Use exactly this template:

```markdown
# Decision Record: <one-line title>

- **Date:** YYYY-MM-DD
- **Status:** Active
- **Source change:** <feature/branch/issue ref, or the prompt that started this>
- **Decision file for Archon:** pass `--input decision_doc=<this-path>`

## Context
<one short paragraph: the problem, the codebase reality, why this matters now>

## Goal
<what counts as success, in the user's terms>

## Decisions
| # | Topic | Decision | Rationale | Source |
|---|-------|----------|-----------|--------|
| 1 | <topic> | <the decision, verbatim where possible> | <why> | User / Delegated |
| 2 | ... | ... | ... | ... |

## Delegated decisions
<Each: the exact question the user delegated, the user's wording ("use your
judgement" is fine), and the decision you therefore own.>

## Explicitly out of scope
<Everything the user said NO to, or that is deferred>

## Validation plan
<How the result will be verified — tests, evals, metrics, manual checks.
Mirror the user's answers from dimension 7.>

## Open Questions
_None._
<!-- If anything remains open, list it here and DO NOT hand this record to a
     planner or execution engine until every item is resolved. -->
```

## Done criteria

- [ ] `## Open Questions` is `_None_` (or everything open was resolved on the spot).
- [ ] Every decision row has a real answer and a Source.
- [ ] You read back the "Decisions" table to the user and they confirmed it.
- [ ] The file path is stated in your reply, so `plan-feature` or the Archon
      `researcher-piv` workflow can be pointed at it.

## Guardrails for later

- If a later step finds the codebase contradicts a recorded decision, STOP and
  re-interview that item rather than bending the decision.
- If you are asked to implement without a decision record, refuse and run this
  skill first. That refusal is the behaviour this pack is built around.
