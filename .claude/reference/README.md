# Reference Documentation

This folder contains **task-specific documentation** that Claude should read only when working on specific areas.

## Purpose

Keep your global rules (`CLAUDE.md`) lightweight by moving detailed patterns here. Claude reads these files only when relevant, protecting the context window.

## How It Works

1. **Global rules** (`CLAUDE.md`) stay short and focused on universal rules
2. **Reference docs** contain deep knowledge for specific task types
3. **Claude reads the right file** when working on that area

## Example Structure

```
.claude/reference/
├── README.md                    # This file
├── frontend-patterns.md         # React/Vue/Angular patterns
├── backend-patterns.md          # API/database patterns
├── testing-patterns.md          # Testing strategies
└── deployment-patterns.md       # CI/CD, Docker, etc.
```

## Template for Reference Docs

```markdown
# [Area] Patterns

## When to Use This Document
Read this when working on [specific area].

## Key Patterns

### Pattern 1: [Name]
[Description and examples]

### Pattern 2: [Name]
[Description and examples]

## Common Gotchas
- [Gotcha 1]
- [Gotcha 2]

## Code Examples

\`\`\`[language]
// Example code
\`\`\`
```

## Referencing in CLAUDE.md

Add a table in your `CLAUDE.md`:

```markdown
## Reference Documentation

| Document | When to Read |
|----------|--------------|
| `.claude/PRD.md` | Understanding project scope and features |
| `.claude/STATUS.md` | Current sprint, priorities, next actions |
| `.claude/reference/frontend-patterns.md` | Building UI components |
| `.claude/reference/backend-patterns.md` | API endpoints, database queries |
```

## Best Practices

1. **Keep it focused** - One topic per file
2. **Include examples** - Code snippets are valuable
3. **Update when learning** - Add patterns as you discover them
4. **Don't duplicate** - Reference external docs when possible
