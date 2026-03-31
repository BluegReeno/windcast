# Path-Scoped Rules (Auto-Loaded)

Rules in this directory are **automatically loaded** when Claude works on files matching their `paths:` frontmatter pattern.

## How It Works

Each rule file has a YAML frontmatter with `paths:` — glob patterns that trigger auto-loading:

```markdown
---
paths:
  - "src/frontend/**/*.tsx"
  - "src/frontend/**/*.ts"
---

# Frontend Conventions

## Tech Stack
- React 19 + Vite + TypeScript
...
```

When Claude opens or edits a file matching `src/frontend/**/*.tsx`, this rule is automatically injected into context. No need to reference it in CLAUDE.md.

## When to Use Rules vs Reference vs Docs

| Location | Auto-loaded? | Size | Use for |
|----------|-------------|------|---------|
| `.claude/rules/` | Yes (by path) | Short (<100 lines) | Conventions, anti-patterns, critical gotchas |
| `.claude/reference/` | No (manual) | Medium | Reusable patterns, code examples |
| `.claude/docs/` | No (sub-agents) | Large (100+ lines) | Deep architecture guides, API references |

## Creating a New Rule

1. Create a `.md` file in this directory
2. Add `paths:` frontmatter with relevant glob patterns
3. Keep it focused: one domain per file
4. Lead with the most critical rules (anti-patterns, gotchas)
5. Keep under 100 lines — if it's longer, split or move to `docs/`

## Example Structure

```
.claude/rules/
├── README.md              # This file
├── frontend.md            # UI framework conventions
├── api.md                 # API endpoint patterns
├── database.md            # Query patterns, migrations
└── testing.md             # Test conventions, mock patterns
```

## Template

See `_template.md` for a starter template.
