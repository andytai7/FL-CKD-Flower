---
name: end-to-end-feature
description: Autonomously develops a complete feature from priming through planning, execution, and commit by chaining the four core PIV-loop skills. Use when you want a full hands-off feature build from a single description.
argument-hint: [feature-description]
---

# End-to-End Feature Development

**Feature Description**: $ARGUMENTS

This skill chains the core PIV-loop skills for feature development. The only
non-negotiable human step is the **design interview**: you are the principal
and no code gets automated against unasked decisions. Everything else here can
run hands-off.

---

## Step 1: Prime - Load Codebase Context

Execute the priming workflow to understand the codebase.

Run the `prime` skill (`.claude/skills/prime/SKILL.md`).

---

## Step 2: Design Interview - Lock the Decisions (MANDATORY, human gate)

Run the `design-interview` skill (`.claude/skills/design-interview/SKILL.md`)
with the same feature description. It interviews you via the `ask` tool and
writes the signed Decision Record to `docs/decisions/<date>-<slug>.md`.

**Do not continue** while its `## Open Questions` is non-empty, and do not
deviate from a recorded decision later — the plan must embed the record as
binding.

---

## Step 3: Planning - Create Implementation Plan

Create a detailed implementation plan for the feature.

Run the `plan-feature` skill (`.claude/skills/plan-feature/SKILL.md`) with the
feature description: **$ARGUMENTS**, pointing it at the Decision Record path
from Step 2.

**IMPORTANT**: Note the feature name (and plan file path) that the planning step creates. You'll need it for the next step.

---

## Step 4: Execute - Implement the Feature

Implement the feature from the plan document.

Run the `execute` skill (`.claude/skills/execute/SKILL.md`) with the plan file path: `.claude/plans/[feature-name].md`.

(Use the feature name from Step 3.)

> Rule-based alternative: instead of this step, run
> `archon workflow run researcher-piv --branch feat/<slug> --input decision_doc=docs/decisions/<date>-<slug>.md "..."`
> for deterministic execution with its own approval gates (see the `archon`
> skill).

---

## Step 5: Commit - Save Changes

Create a git commit for all changes.

Run the `commit` skill (`.claude/skills/commit/SKILL.md`).

---

## Final Summary

After completing all 4 steps, provide:

### Feature Implementation Complete

**Original Request**: $ARGUMENTS

**Feature Name**: [feature-name from planning step]

**Steps Executed:**
1. ✅ Prime - Codebase context loaded
2. ✅ Design Interview - Decision Record locked at `docs/decisions/`
3. ✅ Planning - Plan created at `.claude/plans/[feature-name].md`
4. ✅ Execute - Feature implemented and validated
5. ✅ Commit - Changes committed to git

**Outputs:**
- Plan document: `.claude/plans/[feature-name].md`
- Files created/modified: [list]
- Tests added: [list]
- Commit hash: [hash]

**Next Steps:**
- Push to remote: `git push`
- Create pull request (if applicable)
- Continue with next feature
