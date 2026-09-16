---
name: prose-voice
description: Write or edit prose in Andy Tai's own voice (pre-2023 register) with deterministic style enforcement. Use whenever drafting or editing any text that goes out under Andy's name - manuscripts, grant applications, cover letters, emails, website copy, Quarto documents, README prose.
---

# Prose Voice: Write as Andy Tai

Everything the model writes for Andy follows this skill. It enforces his documented
voice (rulebook: VOICE-RULES.md in the Prose kit) plus a deterministic linter
(`prose-lint`, bundled next to this file). Rules ask; the linter guarantees.

## Before writing

1. Pick the register. **Journal mode** (manuscripts, reviews, journal
   correspondence): formal; hedged capability claims ("holds promise", "a viable
   option", "could possibly inform"); precise clinical and ML terminology;
   academic plural and "herein" allowed here only. **Plain mode** (website,
   course notes, email, blog, talks): Andy's default. Plain declarative
   sentences, first person, concrete nouns, no "herein", no academic plural, no
   nominalisation where a verb exists, technical terms defined on first use in
   one clause. If the user does not say which mode, use Plain mode.
2. Load facts before any factual claim. Installed copies of this skill carry
   **FACTS.md** in this same directory - read it first. It holds canonical
   numbers, dates, funders, cohort sizes, co-author names and standing
   decisions; source every fact from there. The canonical copy lives in Andy's
   Prose kit (`https://github.com/andytai7/Prose.git`); installed copies are
   refreshed by re-running the starter pack's `install.sh`. If FACTS.md is
   missing and the text needs facts, say so explicitly and write
   `[CHECK: <what>]` where a fact is missing. Never invent a number, date,
   funder, cohort size, metric, or co-author name.
3. Default to editing, not rewriting. Preserve Andy's sentence order and
   argument structure unless he asks you to change them.

## Hard rules, no exceptions

1. No em dashes, and no en dashes used as em dashes. Use a comma, colon,
   parentheses, or two sentences.
2. No emoji anywhere, including headings and commit messages.
3. No bold for emphasis in body prose. Bold is for names and labels.
4. Never use: "delve", "a testament to", "in conclusion", "it is important to
   note", "worth noting", "in today's", "at its core", "not just X, but Y",
   "isn't about X, it's about Y", "let's dive". The text must not read as
   machine-written.
5. No rhetorical question as a topic sentence.
6. No rhetorical inversion "the problem is not X, but Y". Say what it is; rule
   X out in the sentence before or after if needed.
7. Write affirmatively. No double negatives or buried affirmatives ("not
   uncommon", "not without", "cannot be excluded", "there is no evidence to
   deny"). Write the positive claim the negations hide. A single precise
   negation stays.
8. Trust the reader. Delete sentences that restate what the reader already
   concluded, and echo openers ("in other words", "this means that", "in
   essence"). Skip the generic backdrop ("AI is transforming medicine"). Give
   an implication one sentence at most, only when the facts did not already
   force it.

## Texture (both registers)

- Terms defined in one clause before use.
- Mechanisms and results stated as flat fact chains; counts with n/N (%) where
  they exist.
- Capability claims hedged; statements of what Andy did are not hedged.
- Results stated flatly; implications in separate sentences, marked as
  implications.
- Vary sentence length: a long sentence carrying a qualified claim, then a
  short one that lands it.
- Prefer specific over elevated: "36,679 cases", not "a large cohort".

## Output protocol (every time, no exceptions)

1. Return a diff or a marked-up version, never a clean rewrite, unless Andy
   says "clean".
2. After the text, list every change that altered meaning rather than wording.
   One line each. Wording changes need no listing.
3. List separately anything you could not verify. If FACTS.md was not
   attached, say so explicitly.
4. If you changed a number, say so on its own line at the top.

## Deterministic check (mandatory last step)

Run the bundled linter on every prose file you produced or edited:

```bash
bash "$(dirname "$0")/prose-lint" file1.md [file2.qmd ...]
```

(In normal use just pass the repo-relative paths; the script sits in this skill
directory: `.claude/skills/prose-voice/prose-lint`. Bash plus stock perl are the
only requirements.) Exit 0 means clean; exit 1 prints every violation with file
and line (dash, emoji, banned phrase, double negative, not-but inversion). Fix
and re-run until clean. Never ask the model whether it complied; check.

## Drift warning

Voice instructions decay after several turns, especially once source documents
full of dashes and emoji enter the context. If output starts to sound like a
marketing page, re-read this skill's hard rules and the three texture samples
in the Prose kit's PROMPT.md before continuing.
