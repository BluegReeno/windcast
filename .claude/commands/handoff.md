---
description: Write a session handoff document for the next agent or session
---

# Handoff: Capture Session State for Continuation

## Objective

Create a structured handoff document that captures everything the next session needs to continue this work. This externalizes session memory into a persistent file, compressed to just the essentials.

## When to Use

- Before ending a long session where work will continue later
- Before hitting context limits (proactive, not reactive)
- When switching from one phase to another (research → implementation)
- When handing off between human and AI sessions

## Process

### 1. Analyze the Current Session

Review everything that happened:
- What was the original goal?
- What has been completed?
- What is still in progress or blocked?
- What key decisions were made and WHY?
- What dead ends were explored?

### 2. Gather Current State

```bash
git status
git diff --stat HEAD
git log --oneline -5
git branch --show-current
```

### 3. Write the Handoff Document

Save to: `HANDOFF.md` in the project root.

**Use this structure:**

```markdown
# Handoff: [Brief Task Description]

**Date:** [current date]
**Branch:** [current branch name]
**Last Commit:** [hash + message, or "uncommitted changes"]

## Goal

[1-2 sentences: what we're trying to accomplish]

## Completed

- [x] [Task 1 — brief description]
- [x] [Task 2 — brief description]

## In Progress / Next Steps

- [ ] [Task 3 — what needs to happen, with enough detail to act on]
- [ ] [Task 4 — include file paths and specific areas]

## Key Decisions

- **[Decision]**: [What was chosen] — [Why, including alternatives rejected]

## Dead Ends (Don't Repeat These)

- [Approach tried] — [Why it failed]

## Files Changed

- `path/to/file.ts` — [what changed and why, 1 line]
- `path/to/new-file.ts` — [NEW: purpose]

## Current State

- **Tests:** [passing/failing — details]
- **Build:** [working/broken]

## Context for Next Session

[2-4 sentences: the MOST IMPORTANT thing the next agent needs to know]

**Recommended first action:** [Exact step to take first]
```

### 4. Confirm and Advise

After writing the handoff:
1. Confirm the file was written
2. Suggest: `Read HANDOFF.md and continue from where the previous session left off.`
3. If uncommitted changes exist, suggest `/commit` first

## Quality Criteria

A good handoff should:
- Let a fresh agent continue without clarifying questions
- Be under 100 lines (concise — link to files, don't duplicate content)
- Include enough "why" that the next agent makes the same decisions
- List dead ends to prevent wasted exploration
- Have a concrete "first action" recommendation

## Anti-patterns

- Don't include full file contents — reference paths instead
- Don't include conversation history — summarize findings
- Don't be vague ("fix the bug") — be specific ("fix X in `path/to/file.ts` by doing Y")
- Don't skip Dead Ends — this prevents the most wasted effort
- Don't skip Key Decisions — without them, the next agent may reverse your choices
