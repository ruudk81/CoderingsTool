# Processor Detection in CoderingsTool

This document lists all utility files that use dynamic processor/CPU count detection for worker pools.

## Files Using Processor Detection

| File | Line | Code | Purpose |
|------|------|------|---------|
| `src/utils/clusterer.py` | 799 | `max_workers = max(1, multiprocessing.cpu_count() - 1)` | UMAP parallel processing |
| `src/utils/clusterer.py` | 801 | `max_workers = multiprocessing.cpu_count()` | Alternative worker count |
| `src/utils/spellChecker.py` | 243 | `pool_size = min(os.cpu_count(), MAX_HUNSPELL_PROCESSES)` | Hunspell process pool |

## Notes

- These files dynamically detect available CPU cores to optimize parallel processing
- On Windows, both `os.cpu_count()` and `multiprocessing.cpu_count()` work the same way
- No changes needed for Windows migration - these are cross-platform compatible

## Related Constants

In `spellChecker.py`:
- `MAX_HUNSPELL_PROCESSES` - Caps the maximum Hunspell processes regardless of CPU count
