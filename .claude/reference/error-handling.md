# Error Handling Patterns

## When to Read
Read this when implementing error handling, logging, or exception management.

---

## Core Principle

Intelligently decide when to fail hard and fast, and when to allow processes to complete despite failures.

---

## When to Fail Fast (Let it Crash!)

These errors should stop execution immediately:

| Category | Examples |
|----------|----------|
| **Startup failures** | Missing credentials, database unreachable, service init failed |
| **Configuration** | Missing env vars, invalid settings |
| **Security** | Auth/authz failures, invalid tokens |
| **Data integrity** | Validation errors, corrupted data, null foreign keys |
| **Critical deps** | Required service unavailable |

**Rule**: Never silently accept bad data. Pydantic should raise, not coerce.

---

## When to Continue but Log

These operations should complete what they can and report failures:

| Category | Behavior |
|----------|----------|
| **Batch processing** | Process all items, report failures at end |
| **Background tasks** | Finish queue, log individual failures |
| **WebSocket events** | Log failed event, continue serving others |
| **Optional features** | Skip if disabled, log and continue |
| **External APIs** | Retry with backoff, then fail with clear message |

**Critical**: Skip failed items entirely — never store corrupted data.

---

## Error Message Guidelines

```python
# Good: Context + IDs + stack trace
logger.error(
    f"Failed to process document {doc_id} from source {source_url}",
    exc_info=True
)

# Bad: Generic message
logger.error("Something went wrong")
```

### Checklist
- [ ] Include what was being attempted
- [ ] Include relevant IDs, URLs, data
- [ ] Preserve stack trace (`exc_info=True`)
- [ ] Use specific exception types (not bare `Exception`)
- [ ] Never return `None` to indicate failure — raise instead
- [ ] For batch ops: report success count AND detailed failure list

---

## Python Example

```python
# Fail fast on startup
def init_database():
    if not os.getenv("DATABASE_URL"):
        raise EnvironmentError("DATABASE_URL is required")

    try:
        connection = connect(os.getenv("DATABASE_URL"))
    except ConnectionError as e:
        raise RuntimeError(f"Cannot connect to database: {e}") from e

    return connection

# Continue but log for batch processing
def process_documents(docs: list[Document]) -> BatchResult:
    successes = []
    failures = []

    for doc in docs:
        try:
            result = process_single(doc)
            successes.append(result)
        except ProcessingError as e:
            logger.error(f"Failed to process {doc.id}: {e}", exc_info=True)
            failures.append({"id": doc.id, "error": str(e)})
            # Continue processing other documents

    return BatchResult(
        success_count=len(successes),
        failure_count=len(failures),
        failures=failures
    )
```
