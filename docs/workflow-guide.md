# Claude Code Project Template

Complete starter template for new projects with Claude Code. Includes battle-tested workflows, progressive context disclosure, and powerful slash commands.

## Contents

### Configuration Files
- **CLAUDE.md** - Global rules and conventions (keep lightweight!)
- **INITIAL.md** - Template for context engineering

### Context System (`.claude/`)

```
.claude/
├── PRD.md           # North star — scope, features, architecture
├── STATUS.md        # Current sprint, priorities, next actions
├── rules/           # Auto-loaded by path pattern (Tier 2)
│   ├── README.md
│   └── _template.md
├── reference/       # Manually loaded patterns (Tier 3)
├── docs/            # Heavy guides for sub-agents (Tier 3)
│   └── README.md
├── agents/          # Specialized sub-agents
├── commands/        # Reusable slash commands
└── tasks/           # Feature tracking files
```

### Slash Commands (`.claude/commands/`)

#### Basic Workflow (simple)
- `/primer` - Load project context
- `/create-plan` - Create implementation plan
- `/execute-plan` - Execute plan with tracking

#### Advanced PIV Workflow (core_piv_loop)
- `/core_piv_loop:prime` - Deep codebase analysis
- `/core_piv_loop:plan-feature` - Detailed planning
- `/core_piv_loop:execute` - Execution with task management

#### Validation & Quality
- `/validation:code-review` - Technical code review
- `/validation:code-review-fix` - Fix bugs found in review
- `/validation:validate` - Complete end-to-end validation
- `/validation:validate-simple` - Quick validation
- `/validation:system-review` - Implementation vs plan analysis
- `/validation:execution-report` - Implementation report

#### Session Management
- `/handoff` - Capture session state for continuation (Write + Compress)
- `/commit` - Create clean commit with AI context tracking

#### GitHub Debugging
- `/github_bug_fix:rca` - Root Cause Analysis of GitHub issue
- `/github_bug_fix:implement-fix` - Implement fix from RCA

#### Utilities
- `/create-prd` - Create Product Requirements Document
- `/end-to-end-feature` - Autonomous complete development

### Specialized Agents (`.claude/agents/`)
- `codebase-analyst.md` - Pattern analysis
- `library-researcher.md` - Library research
- `validator.md` - Testing specialist

---

## Quick Start

### Option 1: GitHub Template (Recommended)

1. Go to https://github.com/bluegreen-ai/project-template
2. Click **"Use this template"** → **"Create a new repository"**
3. Name your new project — all commands are already there!

### Option 2: Manual Clone

```bash
git clone https://github.com/bluegreen-ai/project-template.git my-new-project
cd my-new-project
rm -rf .git
git init
git add .
git commit -m "Initial commit from template"
```

---

## Core Methodology: PRD-First + WISC Context Engineering

This template combines **PRD-First Development** with strategies from the **WISC framework** (Write, Isolate, Select, Compress):

### The Key Techniques

| Technique | WISC Strategy | Implementation |
|-----------|--------------|----------------|
| **PRD-First** | Write | `.claude/PRD.md` is your north star |
| **Path-Scoped Rules** | Select | `.claude/rules/` auto-loaded by file pattern |
| **Progressive Docs** | Select | `reference/` (manual) + `docs/` (sub-agent scouting) |
| **Session Handoff** | Write + Compress | `/handoff` captures state for next session |
| **Context Commits** | Write | `/commit` logs AI context changes in git history |
| **Sub-Agent Isolation** | Isolate | Heavy research delegated to sub-agents |
| **Context Reset** | Compress | `/clear` between planning and execution |

### The 3-Tier Context System

```
Tier 1: CLAUDE.md          → Always loaded, <100 lines of rules
Tier 2: .claude/rules/     → Auto-loaded by path pattern
Tier 3: .claude/docs/      → On-demand via sub-agents
         .claude/reference/ → On-demand, manually referenced
```

**Why?** Loading everything into context causes [context rot](https://research.trychroma.com/context-rot). Progressive disclosure keeps the AI focused on what matters for the current task.

### The Context Reset Pattern

**Critical**: Reset context between planning and execution!

```bash
# 1. Prime and plan
/core_piv_loop:prime
/core_piv_loop:plan-feature "My feature"

# 2. CONTEXT RESET
/clear  # ← Crucial!

# 3. Execute with clean context
/core_piv_loop:execute
```

---

## Workflow Guides

### Simple Workflow (for beginners)

```bash
/primer
/create-plan "Add user authentication"
/execute-plan
/validation:validate-simple
/commit
```

### Advanced PIV Workflow (for serious projects)

```bash
/core_piv_loop:prime
/core_piv_loop:plan-feature "Implement caching system"
# /clear ← context reset
/core_piv_loop:execute
/validation:code-review
/validation:code-review-fix
/validation:validate
/commit
```

### Long Session Workflow

```bash
# ... working for a while ...
# Before ending or hitting context limits:
/handoff
# Next session starts with:
# "Read HANDOFF.md and continue from where the previous session left off."
```

### GitHub Bug Fix Workflow

```bash
/github_bug_fix:rca 42
/github_bug_fix:implement-fix 42
/validation:validate-simple
/commit
```

### Complete Feature Workflow (autonomous)

```bash
/end-to-end-feature "Add CSV export of user data"
```

---

## Creating Path-Scoped Rules

The biggest context engineering win: rules that auto-load only when relevant.

### Example: Frontend Rule

Create `.claude/rules/frontend.md`:

```markdown
---
paths:
  - "src/frontend/**/*.tsx"
  - "src/frontend/**/*.ts"
  - "src/frontend/**/*.css"
---

# Frontend Conventions

## Tech Stack
- React 19 + Vite + TypeScript
- Tailwind CSS v4

## Anti-patterns
- Never use inline styles for theme colors — use Tailwind classes
- Never import from `react-router-dom` — use `react-router` (v7)
```

This loads automatically when Claude touches any `.tsx` file in `src/frontend/`.

See `.claude/rules/README.md` for the full guide.

---

## Resources

- [WISC Framework](https://github.com/coleam00/context-engineering-intro/tree/main/use-cases/ai-coding-wisc-framework)
- [Anthropic: Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Context Rot Research](https://research.trychroma.com/context-rot)
- [Progressive Disclosure for AI Coding](https://alexop.dev/posts/stop-bloating-your-claude-md-progressive-disclosure-ai-coding-tools/)

---

## License

Free to use for your projects.
