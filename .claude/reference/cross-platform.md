# Cross-Platform Standards

## When to Read
Read this when writing shell scripts, CI/CD pipelines, or installation instructions.

---

## Python Command Convention

| Context | Use | Avoid |
|---------|-----|-------|
| Shell scripts | `python3`, `pip3` | `python`, `pip` |
| Shebang | `#!/usr/bin/env python3` | `#!/usr/bin/python` |
| Inside virtualenv | `python`, `pip` (both work) | — |

**Why?** Ubuntu/Linux modern systems don't provide `python` or `pip` by default.

---

## Examples

### Shell Scripts
```bash
#!/bin/bash

# Correct (works on macOS and Linux)
python3 scripts/migrate.py
pip3 install -r requirements.txt

# Incorrect (fails on Ubuntu)
python scripts/migrate.py
pip install -r requirements.txt
```

### Documentation
```markdown
## Installation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Inside venv, both work
pip install -r requirements.txt
python main.py
```

### CI/CD (GitHub Actions)
```yaml
- name: Install dependencies
  run: |
    python3 -m pip install --upgrade pip
    pip3 install -r requirements.txt
```

---

## Other Cross-Platform Notes

### Path Separators
```python
# Use pathlib (cross-platform)
from pathlib import Path
config_path = Path("config") / "settings.json"

# Avoid (Windows issues)
config_path = "config/settings.json"
```

### Line Endings
- Configure `.gitattributes` for consistent line endings
- Use `text=auto` for most files

```gitattributes
* text=auto
*.sh text eol=lf
*.bat text eol=crlf
```

### Environment Variables
```python
# Cross-platform env access
import os
value = os.environ.get("MY_VAR", "default")

# For .env files, use python-dotenv
from dotenv import load_dotenv
load_dotenv()
```
