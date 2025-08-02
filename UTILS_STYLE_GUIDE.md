# CoderingsTool Utils Style Guide

## Module Structure and Organization

### 1. System Path (Line 1)
Every utils module MUST start with this exact line:
```python
import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None
```

### 2. Import Organization (80 "=" separators)

Imports must be organized in the following sections, each with a descriptive header:

```python
# === MODULES ========================================================================================================
# Standard library imports (os, sys, time, asyncio, etc.)
# Third-party general imports (numpy, typing, dataclasses, etc.)

# === MODELS ========================================================================================================
# Pydantic models, BaseModel imports
# Local models imports

# === CONFIG ========================================================================================================
# Configuration imports (config.py items)
# Prompts imports
# Constants imports

# === UTILS ========================================================================================================
# Other util imports (verboseReporter, etc.)
# Special imports (nest_asyncio, etc.)

# === [DOMAIN-SPECIFIC] ===============================================================================================
# Domain-specific imports (e.g., CLUSTERING for themeIdentifier.py)
# Examples: sklearn, umap, hdbscan
```

**Rules:**
- Each section header uses exactly 80 "=" characters after the title
- Section titles in ALL CAPITALS
- Empty line before and after each section
- Within sections: standard library first, then third-party, then local

### 3. Verbose Reporting vs Direct Prints

#### Use VerboseReporter for:
- **Batch process statistics**: Processing counts, rates, timing
- **Debug information**: Detailed processing steps when verbose=True
- **Library warnings/errors**: Automatically captured from tenacity, OpenAI, httpx, etc.
- **Retry attempts**: Special formatting for retry scenarios
- **Rate limiting**: Clear notifications about rate limits

```python
# VerboseReporter with logging capture
from utils.verboseReporter import VerboseReporter

self.verbose_reporter = VerboseReporter(
    enabled=verbose,
    capture_logging=True,  # Automatically captures library warnings
    log_level=logging.WARNING
)

# Regular verbose output
self.verbose_reporter.stat_line("Processed 100 items")

# Library warnings are automatically captured and formatted
# Instead of: WARNING:openai:Rate limit reached...
# You'll see: ⚠️  [openai] Rate limit reached...

# Special methods for common scenarios
self.verbose_reporter.library_retry(1, 3, "Connection timeout", wait_time=2.0)
self.verbose_reporter.rate_limit("OpenAI", wait_time=60.0)
```

#### Use Direct print() for:
- **Main process information**: Critical user-facing status updates
- **Step completion messages**: Major milestones
- **Error messages**: Critical failures that stop execution

```python
print(f"\n'Code assignment' completed in {elapsed_time:.2f} seconds.\n")
print("Error: No cluster results available for code assignment.")
```

### 4. Prompt Printer Usage

Always capture the FIRST prompt in any LLM interaction:

```python
if self.prompt_printer:
    self.prompt_printer.capture_prompt(
        step_name="Step Name",
        utility_name="ClassName",
        prompt_content=prompt,
        prompt_type="Prompt Type"
    )
```

### 5. Class Structure Pattern

```python
class UtilityName:
    """One-line description of the utility."""
    
    def __init__(self, 
                 required_param: str,
                 optional_param: Optional[str] = None,
                 verbose: bool = False,
                 prompt_printer = None):
        self.required_param = required_param
        self.optional_param = optional_param
        self.verbose = verbose
        
        # Use VerboseReporter with logging capture
        self.verbose_reporter = VerboseReporter(
            enabled=verbose,
            capture_logging=True,
            log_level=logging.WARNING
        )
        self.prompt_printer = prompt_printer
        
        # Initialize any clients or configs
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Initial status if verbose
        self.verbose_reporter.stat_line(f"Initialized {self.__class__.__name__}")
```

### 6. Async Pattern

```python
# At module level after imports
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# In class
async def _process_async(self) -> Result:
    """Private async method"""
    pass

def process(self) -> Result:
    """Public sync wrapper"""
    return asyncio.run(self._process_async())
```

### 7. Error Handling and Logging

```python
# Option 1: Use VerboseReporter with logging capture (RECOMMENDED)
from utils.verboseReporter import VerboseReporter

class MyUtility:
    def __init__(self, verbose: bool = False):
        # This automatically captures library warnings/errors
        self.verbose_reporter = VerboseReporter(
            enabled=verbose,
            capture_logging=True,  # Captures warnings from openai, httpx, etc.
            log_level=logging.WARNING
        )

# Option 2: Manual logging suppression (if not using enhanced reporter)
import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use tenacity for retries
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
async def api_call(self):
    # With VerboseReporter, retry warnings will show as:
    # 🔄 Retry 1/3: Rate limit error (waiting 4.0s)
    pass
```

### 8. Constants and Configuration

- Use UPPERCASE for module-level constants
- Import from config.py when possible
- Define at module level after imports:

```python
EMBEDDING_DIMENSION = 3072  # text-embedding-3-large dimension
DEFAULT_BATCH_SIZE = 100
```

## Template for New Utils

```python
import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import logging
from typing import List, Optional

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY
from prompts import RELEVANT_PROMPT

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Module constants
DEFAULT_VALUE = 100


class NewUtility:
    """Brief description of what this utility does."""
    
    def __init__(self, 
                 required_param: str,
                 verbose: bool = False,
                 prompt_printer = None):
        self.required_param = required_param
        self.verbose = verbose
        
        # VerboseReporter with automatic logging capture
        self.verbose_reporter = VerboseReporter(
            enabled=verbose,
            capture_logging=True,  # Captures warnings from libraries
            log_level=logging.WARNING
        )
        self.prompt_printer = prompt_printer
        
        self.verbose_reporter.stat_line(f"Initialized {self.__class__.__name__}")
    
    async def _process_async(self) -> models.SomeModel:
        """Core async processing logic."""
        # Main process information - use print()
        print(f"Processing {self.required_param}...")
        
        # Batch stats - use verbose reporter
        self.verbose_reporter.stat_line(f"Processed {count} items")
        
        # Library warnings are automatically captured and formatted
        # No need for manual logging configuration
        
        return result
    
    def process(self) -> models.SomeModel:
        """Public interface for async processing."""
        return asyncio.run(self._process_async())
```

## Key Principles

1. **Consistency**: All utils follow the same structure
2. **Clarity**: Section headers make navigation easy
3. **Verbosity Control**: Users control output detail level
4. **Error Visibility**: Critical errors always visible
5. **Prompt Transparency**: First prompts always captured
6. **Async-First**: Use async patterns for I/O operations