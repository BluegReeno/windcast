---
description: Execute a development plan with TASKS.md file-based task tracking
argument-hint: [plan-file-path]
---

# Execute Development Plan with TASKS.md Tracking

You are about to execute a comprehensive development plan with file-based task tracking. This workflow ensures systematic task tracking and implementation throughout the entire development process.

## Critical Requirements

**MANDATORY**: Throughout the ENTIRE execution of this plan, you MUST maintain continuous usage of TASKS.md for task management. Every task from the plan must be tracked from creation to completion using markdown files.

## Step 1: Read and Parse the Plan

Read the plan file specified in: $ARGUMENTS

The plan file will contain:
- A list of tasks to implement
- References to existing codebase components and integration points
- Context about where to look in the codebase for implementation

## Step 2: Create Task File

1. Create a task file at `.claude/tasks/{feature-name}.md`
2. Use the template from `.claude/tasks/_template.md`
3. Populate with tasks from the plan
4. Update `.claude/STATUS.md` to point to this task file

## Step 3: Populate Task File

Extract ALL tasks from the plan and add them to the task file:

```markdown
## Tasks

### Phase 1: {Phase Name}
- [ ] Task 1 from plan
- [ ] Task 2 from plan

### Phase 2: {Phase Name}
- [ ] Task 3 from plan
...
```

Ensure tasks are ordered by dependency (top-to-bottom execution).

## Step 4: Codebase Analysis

Before implementation begins:
1. Analyze ALL integration points mentioned in the plan
2. Use Grep and Glob tools to:
   - Understand existing code patterns
   - Identify where changes need to be made
   - Find similar implementations for reference
3. Read all referenced files and components
4. Build a comprehensive understanding of the codebase context

## Step 5: Implementation Cycle

For EACH task in sequence:

### 5.1 Start Task
- Mark the current task as in-progress in the task file:
  ```
  - [ ] @claude Task description
  ```
- Use the Edit tool to make this change
- Use TodoWrite to track local subtasks if needed

### 5.2 Implement
- Execute the implementation based on:
  - The task requirements from the plan
  - Your codebase analysis findings
  - Best practices and existing patterns
- Make all necessary code changes
- Ensure code quality and consistency

### 5.3 Complete Task
- Mark the task as complete with date:
  ```
  - [x] Task description ✓ 2026-01-17
  ```
- Use the Edit tool to make this change
- Proceed to next task

### 5.4 Proceed to Next
- Move to the next task in the list
- Repeat steps 5.1-5.3

**CRITICAL**: Only ONE task should be in progress at any time. Complete each task before starting the next.

## Step 6: Validation Phase

After ALL tasks are complete:

**IMPORTANT: Use the `validator` agent for comprehensive testing**
1. Launch the validator agent using the Task tool
   - Provide the validator with a detailed description of what was built
   - Include the list of features implemented and files modified
   - The validator will create simple, effective unit tests
   - It will run tests and report results

The validator agent will:
- Create focused unit tests for the main functionality
- Test critical edge cases and error handling
- Run the tests using the project's test framework
- Report what was tested and any issues found

Additional validation you should perform:
- Check for integration issues between components
- Ensure all acceptance criteria from the plan are met

## Step 7: Finalize

After successful validation:

1. Ensure all tasks are marked `[x]` in the task file
2. Add completion date to the task file header:
   ```markdown
   ## Completion
   - **Started**: 2026-01-17
   - **Completed**: 2026-01-17
   - **Commit**: (link to commit when done)
   ```
3. Move task file to `.claude/tasks/_archive/` if feature is fully complete
4. Update `.claude/STATUS.md` with next focus

## Step 8: Final Report

Provide a summary including:
- Total tasks created and completed
- Test coverage achieved
- Key features implemented
- Any issues encountered and how they were resolved

## Workflow Rules

1. **NEVER** skip task tracking at any point
2. **ALWAYS** create task file before starting implementation
3. **MAINTAIN** one task in progress at a time (marked with `@claude`)
4. **VALIDATE** all work before marking feature as complete
5. **TRACK** progress continuously through task file updates
6. **ANALYZE** the codebase thoroughly before implementation
7. **TEST** everything before final completion

## Notes

- If you encounter issues not addressed in the plan, document them in the Notes section
- If you need to deviate from the plan, explain why in the task file
- If tests fail, fix implementation until they pass
- Don't skip validation steps

### Ready for Commit
- Confirm all changes are complete
- Confirm all validations pass
- Ready for `/commit` command
