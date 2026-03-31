# Deep Reference Documentation

This folder contains **heavy reference guides** (100+ lines) that are NOT auto-loaded into context.

## How It Works

These documents are designed for the **sub-agent scouting pattern**:

1. A sub-agent reads the document header to determine relevance
2. Only if relevant, the full document is loaded
3. The main session context stays clean

This keeps large architecture guides, API references, and implementation walkthroughs out of your primary context window.

## When to Use Docs vs Rules vs Reference

| Location | Auto-loaded? | Size | Use for |
|----------|-------------|------|---------|
| `.claude/rules/` | Yes (by path) | Short (<100 lines) | Conventions, anti-patterns, critical gotchas |
| `.claude/reference/` | No (manual) | Medium | Reusable patterns, code examples |
| `.claude/docs/` | No (sub-agents) | Large (100+ lines) | Deep architecture guides, API references |

## Document Format

Start every doc with a clear header that lets a sub-agent decide relevance without reading the full file:

```markdown
# [Topic] Deep Dive

> **Scope:** [What this covers]
> **Read when:** [Specific trigger — e.g., "implementing a new API adapter"]
> **Size:** ~[N] lines

## Overview
[2-3 sentence summary]

## [Detailed sections...]
```

## Example Structure

```
.claude/docs/
├── README.md                        # This file
├── architecture-deep-dive.md        # Full system architecture
├── api-reference.md                 # Complete API documentation
└── deployment-guide.md              # Infrastructure and CI/CD
```
