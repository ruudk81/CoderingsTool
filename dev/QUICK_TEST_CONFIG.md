# Quick Test Configuration

For rapid validation after code changes without burning excessive resources.

## Standard Test Parameters

```python
QUICK_TEST_CONFIG = {
    "filename": "M000000 Associatiemonitor Merk X net databestand.sav",
    "id_column": "DLNMID",
    "var_name": "Qd1_combined",
    "sample_size": 20,  # Small sample for fast testing
}
```

## When to Use

- ✅ After refactoring or code changes
- ✅ Before committing changes
- ✅ Testing new features
- ✅ Verifying bug fixes
- ❌ NOT for final validation (use full dataset)

## Benefits

| Aspect | Quick Test | Full Dataset |
|--------|-----------|--------------|
| **Time** | Minutes | Hours |
| **Cost** | ~$0.10-0.50 | ~$5-20 |
| **Sample Size** | 20 responses | 2000 responses |
| **Purpose** | Rapid validation | Production testing |

## Usage

### Environment Variable Method
```bash
export SAMPLE_SIZE=20
cd src && python pipeline.py
```

### Direct Pipeline Modification
Edit `src/pipeline.py` temporarily:
```python
sample_size = 20  # Quick test
```

## What Gets Tested

With sample_size=20, the pipeline will:
- Load 20 responses
- Run all 9 processing steps
- Generate embeddings (~20 API calls)
- Create small clusters
- Generate test codebook
- Complete in ~2-5 minutes

## Warning

⚠️ **This is NOT a substitute for full testing!**

Always run with full dataset before:
- Pushing to production
- Delivering results to clients
- Making critical decisions

## Location

This config is also documented in:
- `CLAUDE.md` (main project instructions)
- This file (`dev/QUICK_TEST_CONFIG.md`)

Keep both in sync if values change.

---

**Last Updated:** 2025-10-23
