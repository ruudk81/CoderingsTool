# Pipeline Step Migration Guide

This guide documents the strategy for migrating experimental implementations to production in the CoderingsTool pipeline.

## Overview

The migration pattern follows a consistent approach:
1. Develop improvements in `src/experiments/{step}_v{N}/`
2. Test thoroughly via experiment runner
3. Migrate to production with step-specific config
4. Archive old version for rollback capability
5. Update pipeline and experiment runner imports

---

## Migration Pattern

### Directory Structure

```
src/
├── config.py                    # Main config (unchanged)
├── config_{step}.py             # NEW: Step-specific config
├── pipeline.py                  # Update imports
├── prompts.py                   # Add new prompts if needed
├── utils/
│   ├── {step}.py                # REPLACE: Production implementation
│   ├── {step}_helpers.py        # NEW: If helpers needed (optional)
│   └── old/
│       └── {step}_v{N-1}.py     # ARCHIVE: Previous version
└── experiments/
    └── {step}_v{N}/
        ├── __init__.py          # UPDATE: Re-export from production
        ├── run_experiment.py    # UPDATE: Toggle imports
        ├── {step}.py            # DELETE: After migration
        ├── config.py            # DELETE: After migration
        └── prompts.py           # DELETE: After migration (if exists)
```

### Naming Conventions

| Component | Experiment | Production |
|-----------|------------|------------|
| Class | `{Step}V{N}` (e.g., `ClustererV3`) | `{Step}` (e.g., `Clusterer`) |
| Config | `{Step}V{N}Config` | `{Step}Config` |
| File | `experiments/{step}_v{N}/{step}.py` | `utils/{step}.py` |
| Config file | `experiments/{step}_v{N}/config.py` | `config_{step}.py` |

---

## Step-by-Step Migration Process

### Phase 1: Archive Current Production

```bash
mkdir -p src/utils/old/
cp src/utils/{step}.py src/utils/old/{step}_v{N-1}.py
```

**Why:** Preserves rollback capability. Never delete the old version until migration is fully verified.

### Phase 2: Create Step-Specific Config

Create `src/config_{step}.py`:

```python
"""
{Step}-specific configuration - separate from main config.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class {Step}Config:
    """Configuration for {Step}."""

    # Group 1: Core settings
    param1: str = "default"
    param2: int = 10

    # Group 2: Feature flags
    enable_feature: bool = True

    # Group 3: Output
    verbose: bool = True

# Default instance
DEFAULT_{STEP}_CONFIG = {Step}Config()

# Preset configurations (optional)
FAST_{STEP}_CONFIG = {Step}Config(
    enable_feature=False,
)
```

**Best practices:**
- Use dataclass for clean parameter organization
- Group related parameters with comments
- Provide sensible defaults matching approved experiment settings
- Include preset configurations for common use cases

### Phase 3: Migrate Implementation Files

1. **Copy main implementation:**
   ```bash
   cp src/experiments/{step}_v{N}/{step}.py src/utils/{step}.py
   ```

2. **Copy helpers (if any):**
   ```bash
   cp src/experiments/{step}_v{N}/{step}_helpers.py src/utils/{step}_helpers.py
   ```

3. **Update imports in production files:**
   ```python
   # Change FROM (experiment paths):
   from .config import {Step}V{N}Config
   from .{step}_helpers import helper_func

   # Change TO (production paths):
   from config_{step} import {Step}Config
   from utils.{step}_helpers import helper_func
   ```

4. **Rename classes:**
   - `{Step}V{N}` → `{Step}`
   - `{Step}V{N}Config` → `{Step}Config`

5. **Remove dev artifacts:**
   - Delete `sys.path.insert(...)` lines (for path hacking)
   - Delete `if __name__ == "__main__":` test blocks
   - Keep necessary imports like `os`, `sys` if used elsewhere!

### Phase 4: Update Prompts (if applicable)

Add to `src/prompts.py`:

```python
# Keep old prompt for reference
{STEP}_PROMPT_V{N-1} = """..."""

# New prompt (active)
{STEP}_PROMPT = {STEP}_PROMPT_V{N} = """..."""

# Pydantic model for structured output (if needed)
class {Step}Response(BaseModel):
    field1: str = Field(...)
    field2: List[str] = Field(...)
```

### Phase 5: Update Pipeline Integration

Update `src/pipeline.py`:

```python
# Update imports
from utils.{step} import {Step}
from config_{step} import {Step}Config

def step_{N}_{step}(...):
    # Create config
    config = {Step}Config(
        param1="value",
        verbose=verbose,
    )

    # Instantiate with config
    processor = {Step}(data, config=config)
    result = processor.run()

    # Cache results
    cache_manager.save_to_cache(...)
```

### Phase 6: Update Experiment Runner

Update `src/experiments/{step}_v{N}/run_experiment.py`:

```python
"""
{Step} V{N} Experiment Runner

NOTE: After migration, v{N} is now the PRODUCTION {step}.
The old {step} is archived at src/utils/old/{step}_v{N-1}.py.

Toggle modes:
    USE_EXPERIMENTAL = True  -> Uses PRODUCTION (migrated v{N})
    USE_EXPERIMENTAL = False -> Uses OLD (archived v{N-1})
"""

USE_EXPERIMENTAL = True

if USE_EXPERIMENTAL:
    from utils.{step} import {Step}
    from config_{step} import {Step}Config
else:
    from utils.old.{step}_v{N-1} import {Step}V{N-1} as {Step}
```

Update `src/experiments/{step}_v{N}/__init__.py`:

```python
"""
{Step} V{N} Module

NOTE: After migration, v{N} is now the PRODUCTION {step}.
Import from utils.{step} and config_{step} instead.

This module provides backward-compatible aliases.
"""

from utils.{step} import {Step}
from config_{step} import {Step}Config

# Backward-compatible aliases
{Step}V{N} = {Step}
{Step}V{N}Config = {Step}Config

__all__ = ["{Step}", "{Step}Config", "{Step}V{N}", "{Step}V{N}Config"]
```

### Phase 7: Test via Pipeline

```bash
cd src && python pipeline.py
```

With `FORCE_RECALCULATE_ALL = True` or specific step forced.

**Check:**
- [ ] No import errors
- [ ] Step runs without exceptions
- [ ] Output matches expected format
- [ ] Cache saves correctly

### Phase 8: Test via App

```bash
streamlit run src/app.py --server.headless true
```

**Check:**
- [ ] Step page loads
- [ ] Step executes correctly
- [ ] Results display properly
- [ ] Cache recovery works

### Phase 9: Cleanup and Commit

1. **Delete migrated experiment files:**
   ```bash
   rm src/experiments/{step}_v{N}/{step}.py
   rm src/experiments/{step}_v{N}/{step}_helpers.py  # if exists
   rm src/experiments/{step}_v{N}/config.py
   rm src/experiments/{step}_v{N}/prompts.py  # if exists
   ```

2. **Keep:**
   - `run_experiment.py` (updated with toggle)
   - `__init__.py` (updated to re-export from production)

3. **Commit:**
   ```bash
   git add src/config_{step}.py
   git add src/utils/{step}.py
   git add src/utils/{step}_helpers.py  # if exists
   git add src/prompts.py
   git add src/pipeline.py
   git add src/experiments/{step}_v{N}/

   git commit -m "feat: Migrate {step}_v{N} to production with step-specific config

   - Add {Step}Config to src/config_{step}.py
   - Migrate {Step} + helpers to src/utils/
   - Update pipeline step_{N} with new imports
   - Update experiment runner with production/archived toggle

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
   ```

---

## Gotchas and Lessons Learned

### 1. Don't Remove Standard Library Imports

**Problem:** Removing `import os` and `import sys` thinking they were dev artifacts, but they were used elsewhere in the file.

**Symptom:**
```
NameError: name 'os' is not defined
```

**Solution:** Only remove `sys.path.insert(...)` lines, not the imports themselves. Check if `os`, `sys`, etc. are used elsewhere before removing.

### 2. Update Relative Import Paths

**Problem:** Relative imports like `from .config import ...` or `from representation.module import ...` break when files move.

**Symptom:**
```
ModuleNotFoundError: No module named 'representation'
```

**Solution:** Update all imports to absolute paths:
```python
# FROM (experiment):
from .config import Config
from representation.ctfidf import CTfidf

# TO (production):
from config_{step} import Config
from experiments.representation.ctfidf import CTfidf
```

### 3. Helpers File Import Chain

**Problem:** Helper files also have imports that need updating.

**Solution:** Update imports in BOTH main file AND helpers file:
- Main file: `from utils.{step}_helpers import ...`
- Helpers file: `from config_{step} import ...`, `from prompts import ...`

### 4. Config Parameter Mismatch

**Problem:** Experiment config has different parameter names than expected.

**Solution:**
- Review experiment's config.py carefully
- Map all parameters to new config dataclass
- Use approved experiment defaults as production defaults

### 5. Cache Key Compatibility

**Problem:** Changed cache structure can break existing caches.

**Solution:**
- Keep cache step names consistent (`"initial_clusters"`, not `"clusters_v3"`)
- If adding new cache layers, use new step names
- Document cache changes in commit message

### 6. Streamlit Headless Mode

**Problem:** Streamlit hangs at ~10% when trying to auto-open browser on macOS.

**Solution:**
```bash
streamlit run src/app.py --server.headless true
```

Or add to `~/.streamlit/config.toml`:
```toml
[server]
headless = true
```

---

## Migration Checklist

Use this checklist for each migration:

### Pre-Migration
- [ ] Experiment thoroughly tested via `run_experiment.py`
- [ ] All features working as expected
- [ ] Config parameters documented

### Migration
- [ ] Phase 1: Archive old version to `utils/old/`
- [ ] Phase 2: Create `config_{step}.py`
- [ ] Phase 3: Copy and update implementation files
- [ ] Phase 4: Update `prompts.py` (if applicable)
- [ ] Phase 5: Update `pipeline.py` imports
- [ ] Phase 6: Update experiment runner with toggle
- [ ] Phase 7: Test via `pipeline.py`
- [ ] Phase 8: Test via `app.py`
- [ ] Phase 9: Delete experiment files, commit

### Post-Migration
- [ ] Verify rollback works (set toggle to False)
- [ ] Push to remote when ready

---

## Completed Migrations

| Step | Experiment | Production | Date | Commit |
|------|------------|------------|------|--------|
| Step 3: Idea Extraction | `ideaExtractor_v3` | `utils/ideaExtractor.py` | 2026-01 | `043ef99` |
| Step 4: Embeddings | `embedder_v2` | `utils/embedder.py` | 2026-01 | `9b2fc6b` |
| Step 5: Clustering | `clusterer_v3` | `utils/clusterer.py` | 2026-01 | `d31f507` |

---

## Rollback Procedure

If issues arise after migration:

### Option A: Toggle to Old Version
Set `USE_EXPERIMENTAL = False` in experiment runner and update pipeline imports:
```python
from utils.old.{step}_v{N-1} import {Step}
```

### Option B: Git Revert
```bash
git revert HEAD
```

### Option C: Manual Restore
```bash
cp src/utils/old/{step}_v{N-1}.py src/utils/{step}.py
rm src/utils/{step}_helpers.py  # if exists
# Update pipeline.py imports back to old class names
```

---

## Future Improvements

Consider for future migrations:
1. Automated migration script
2. Unit tests for migrated components
3. Integration tests comparing old vs new output
4. Performance benchmarks before/after migration