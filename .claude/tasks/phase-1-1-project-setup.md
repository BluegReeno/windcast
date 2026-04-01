# Feature: Phase 1.1 — Project Setup

## Goal
Bootstrap the WindCast Python project with all dependencies, tooling, and package structure.

## Context
- **PRD Reference**: `.claude/PRD.md` lines 136-198 (directory structure), 328-395 (tech stack)
- **Plan**: `.claude/plans/phase-1-1-project-setup.md`

## Tasks

### Phase 1: Project Configuration
- [x] Create `pyproject.toml` with dependencies and tool configs ✓ 2026-04-01
- [x] Create `src/windcast/` package structure with `__init__.py` files ✓ 2026-04-01
- [x] Update `.gitignore` for Python/data/MLflow/venv ✓ 2026-04-01

### Phase 2: Environment Setup
- [ ] @claude Run `uv sync` to install dependencies and generate lockfile

### Phase 3: Validation
- [x] Create smoke test `tests/test_setup.py` ✓ 2026-04-01
- [x] Run full validation suite (ruff, pyright, pytest) ✓ 2026-04-01

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | Create | Project config, deps, tool settings |
| `src/windcast/__init__.py` | Create | Root package |
| `src/windcast/data/__init__.py` | Create | Data sub-package |
| `src/windcast/features/__init__.py` | Create | Features sub-package |
| `src/windcast/models/__init__.py` | Create | Models sub-package |
| `tests/__init__.py` | Create | Tests package |
| `tests/test_setup.py` | Create | Smoke tests |
| `scripts/.gitkeep` | Create | Scripts placeholder |
| `.gitignore` | Modify | Add Python/data/MLflow entries |

## Notes

## Completion
- **Started**: 2026-04-01
- **Completed**: 2026-04-01
- **Commit**: (pending `/commit`)
