# Project Guidelines for Claude Code

## Language Policy

- **Conversations**: French OK
- **Code, docs, commits**: English only

---

## Project Overview

**[Project Name]** - [One sentence description]

**Full specifications**: See `.claude/PRD.md`

---

## Tech Stack

- **Language**: [e.g., TypeScript, Python]
- **Framework**: [e.g., Next.js, FastAPI]
- **Database**: [e.g., PostgreSQL, Supabase]
- **Testing**: [e.g., Jest, pytest]

---

## Context System (3 tiers)

Context is organized in progressive disclosure — load only what you need:

### Tier 1: Global Rules (this file)

Always loaded. Keep lean (<100 lines of actual rules). If removing a line wouldn't cause mistakes, cut it.

### Tier 2: Path-Scoped Rules (`.claude/rules/`)

Auto-loaded when you work on matching files. Each rule has a `paths:` frontmatter with glob patterns.

Example: `.claude/rules/frontend.md` loads when editing `src/frontend/**/*.tsx`.

### Tier 3: Reference & Deep Docs

| Location | When to read |
|----------|--------------|
| `.claude/reference/` | Manually, for reusable patterns and code examples |
| `.claude/docs/` | Via sub-agents, for heavy guides (100+ lines) |

### Always-Available Documents

| Document | When to Read |
|----------|--------------|
| `.claude/PRD.md` | Project scope, features, architecture |
| `.claude/STATUS.md` | Current sprint, priorities, next actions |

---

## Task Management

**Workflow:**
1. Check `STATUS.md` for current focus
2. Read active task file in `.claude/tasks/`
3. Mark `@claude` when starting, `[x] ✓ YYYY-MM-DD` when done
4. Move completed features to `_archive/`

---

## Code Style

### Naming
- Files: `kebab-case.ts` / `snake_case.py`
- Classes: `PascalCase`
- Functions: `camelCase` (JS) / `snake_case` (Python)
- Constants: `UPPER_SNAKE_CASE`

### Imports
- Group: stdlib → third-party → local
- Sort alphabetically within groups

### Formatting
- Use project formatter (Prettier/Black)
- Max line length: 100 chars

---

## Core Principles

- **Fix forward** — no backward compatibility, remove deprecated code immediately
- **Fail fast** — detailed errors over graceful failures
- **KISS / DRY / YAGNI** — simple, no repetition, no overbuilding
- **Clean comments** — describe functionality, not changes (avoid "LEGACY", "REMOVED", "SIMPLIFIED")

---

## Testing

- Test files: `*.test.ts` / `test_*.py`
- Run before commit
- Prefer integration tests over unit tests for APIs

---

## Session Management

- Use `/handoff` before ending long sessions to capture state for the next session
- Use `/commit` with the `Context:` section when AI context files change

---

## Common Gotchas

<!-- Customize for your project -->
- Update schema when adding database fields
- Use parameterized queries (never string concatenation)
- Check for null before accessing nested properties

---

## External Resources

- [PRD](.claude/PRD.md) | [Status](.claude/STATUS.md) | [README](README.md)
