# CoderingsTool Utils Style Guide

## Key Requirements

### 1. System Path (Line 1)
Every utils module MUST start with this exact line:
```python
import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None
```

### 2. Import Organization
Organize imports with clear section headers using 80 "=" characters:

```python
# === MODULES ========================================================================================================
# Standard library imports (time, asyncio, typing, etc.)
# Third-party imports (numpy, pandas, pydantic, etc.)

# === MODELS ========================================================================================================
# Pydantic models, BaseModel imports
# Local models imports

# === CONFIG ========================================================================================================
# Configuration imports (config.py items)
# Prompts imports

# === UTILS ========================================================================================================
# VerboseReporter and other util imports
# Special imports (nest_asyncio, etc.)
```

### 3. VerboseReporter Usage
- Use VerboseReporter with logging capture enabled
- Standard initialization: `VerboseReporter(verbose, capture_logging=True)`
- Remove manual logging suppression (let VerboseReporter handle it)

### 4. Class Structure Pattern
```python
class UtilityName:
    def __init__(self, 
                 required_param: str,
                 verbose: bool = False,
                 prompt_printer = None):
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        # ... other initialization
```

### 5. Prompt Capture
Always capture the FIRST prompt in LLM interactions:
```python
if self.prompt_printer:
    self.prompt_printer.capture_prompt(
        step_name="Step Name",
        utility_name="ClassName", 
        prompt_content=prompt,
        prompt_type="Prompt Type"
    )
```

### 6. Async Patterns
```python
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass
```

### 7. No Manual Logging Suppression
Remove lines like:
```python
logging.getLogger("openai").setLevel(logging.WARNING)
```
Let VerboseReporter handle library logging instead.