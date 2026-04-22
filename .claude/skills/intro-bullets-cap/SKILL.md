---
name: intro-bullets-cap
description: Audit a lecture or example notebook for long prose passages that could be revised into the "intro sentence → bullets → cap sentence" format, and report candidates without editing.
disable-model-invocation: true
allowed-tools: Read Glob Grep
argument-hint: [notebook-path]
---

Read the notebook at $ARGUMENTS (or the notebook the user indicates).

Scan every markdown cell (including the bodies of blockquotes) and identify prose passages that would read better in the **intro → bullets → cap** style:

- **Intro sentence:** one short sentence that frames what follows.
- **Bullets (1–3):** each bullet is `* __[Title]:__ [1–2 sentences of meat]`. Bullets must have substance, not just labels.
- **Cap sentence:** one short sentence that closes the idea.

## What to flag

A passage is a candidate if **any** of the following hold:

1. **Length:** 4 or more sentences of continuous prose on a single theme, with no lists or equations breaking it up.
2. **Enumerable structure:** The text already lists 2–3 distinct facts, reasons, contrasts, or components in sentence form (watch for "first... second...", "because... and also...", "one is... another is...", or comma-separated parallel clauses that could be bullets).
3. **Dense blockquote:** A text-only blockquote whose body is a single paragraph of 3+ sentences, especially when the sentences are doing different jobs (setup, mechanism, contrast, implication).
4. **Mixed-purpose paragraph:** One paragraph that introduces a concept, explains it, contrasts it with something else, and concludes, all run together.

Do **not** flag:

- Short asides (1–3 sentences).
- Passages that are already well-structured (existing bullets, tables, algorithm blocks).
- Derivation paragraphs where the sentence-to-sentence flow is the point (step-by-step math explanations).
- Definition or theorem blockquotes whose form is dictated by mathematical content.

## What to report

For each candidate, report:

1. **Location:** cell id (or section heading) and the first few words of the passage.
2. **Why it's a candidate:** one line citing which of the criteria above it meets.
3. **Proposed split:** a short sketch of the intro sentence, 1–3 bullet titles (no full bodies needed), and the cap sentence. Enough to show the passage decomposes cleanly; not a full rewrite.

Order candidates by how much the revision would improve readability (strongest candidates first).

## What not to do

- Do not edit the notebook. This skill is audit-only.
- Do not suggest bullets of fewer than one full sentence. Thin bullets are worse than prose.
- Do not force a passage into the format if it does not decompose cleanly into 1–3 substantive bullets. Say so and skip it.
- Do not run the full CLAUDE.md review workflow. This is a targeted formatting pass.
