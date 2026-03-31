# [Project Name]

[One sentence description of your project]

## Quick Start

```bash
# Install dependencies
[your install command]

# Run development server
[your dev command]

# Run tests
[your test command]
```

## Project Structure

```
src/
├── ...
```

## Documentation

- **[PRD](.claude/PRD.md)** - Product Requirements Document
- **[Status](.claude/STATUS.md)** - Current sprint and priorities
- **[Workflow Guide](docs/workflow-guide.md)** - Development methodology and commands

## AI Context Architecture

This project uses a **3-tier context system** for efficient AI-assisted development:

| Tier | Location | Loaded | Purpose |
|------|----------|--------|---------|
| 1 | `CLAUDE.md` | Always | Global rules, tech stack, core principles |
| 2 | `.claude/rules/` | Auto (by path) | Domain-specific conventions |
| 3 | `.claude/docs/` | On-demand | Deep architecture guides |

See `.claude/rules/README.md` and `.claude/docs/README.md` for details.

## Development

This project uses Claude Code with a PRD-first methodology. See the [workflow guide](docs/workflow-guide.md) for details.

### Key Commands

```bash
/core_piv_loop:prime          # Load project context
/core_piv_loop:plan-feature   # Plan a new feature
/core_piv_loop:execute        # Execute with task tracking
/handoff                      # Capture session state for continuation
/commit                       # Create clean commit with context tracking
```

## License

[Your license]
