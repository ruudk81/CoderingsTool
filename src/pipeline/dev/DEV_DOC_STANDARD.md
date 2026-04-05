# Dev Doc Standard

Standard structure for the 4 markdown files in each step's `dev/` folder. Step 1 PreProcessor is the reference implementation.

## Ownership rules

Each file has a single responsibility. No duplication across files.

| Topic | Owner | NOT in |
|-------|-------|--------|
| Purpose, key files, gotchas | CLAUDE.md | ARCHITECTURE.md |
| Input/output models, cache keys, data flow | CACHE_LOGIC.md | ARCHITECTURE.md |
| Pipeline stages, prompt details, config tables | ARCHITECTURE.md | — |
| Runtime behavior, rate limiting, known issues | PROCESSING.md | — |

---

## CLAUDE.md — Entry point

The first file Claude reads when familiarizing with a step. Concise overview, no deep detail.

```
# Step N — Name

## Purpose
One paragraph.

## Key Files
- `file.py` — what it does

## Input / Output Contract
- **Input**: model + cache key
- **Output**: model + cache key + what's added

## LLM Usage
Model tier, prompt file, response model, dispatch pattern.
(or "None" for non-LLM steps)

## Shared Utils
Which utils/ modules this step depends on.

## Gotchas
Bullet list of non-obvious behavior and pitfalls.

## Processing Phases
(only for LLM steps)
Numbered list of high-level phases.

## Dev Docs
Links to the other 3 files.
```

---

## ARCHITECTURE.md — System design

The "what and why". Structure, design choices, prompts, concurrency approach, configuration.

```
# Step N — Architecture

## Design Intent
What the step does + key design choices (bullet list).

## Pipeline Overview
ASCII flow diagram showing stages from input to output.

## Prompt Builder & Response Model
Table of prompts, response models, and notes.
Detail on prompt input/output structure.

## Concurrency & Rate Limiting
Stack overview: what layers, what dispatch pattern.
Subsections per processing domain (e.g., Hunspell vs LLM).

## Configuration
Tables of config dataclass fields + module-level constants.

## [Step-specific sections]
e.g., "Inverted Index Optimization" — only if the step has
notable implementation patterns worth documenting.
```

**Does NOT contain**: data flow diagrams, model definitions, file lists (those live in CACHE_LOGIC.md and CLAUDE.md).

---

## CACHE_LOGIC.md — Data contracts

The "what goes in, what comes out, how it's stored".

```
# Step N — Cache Logic

## Cache Type
Growing model vs metadata. Storage format.

## Input Files
Table: step name, prefix, model class, type, method, contents.

## Output Files
Same table format.

## Cache Key Scheme
Variable key generation, file naming pattern, SQLite key.

## Cache-Hit Logic
Code snippet showing the check + force_recalc note.

## Growing Model Assembly
How the output list is built (numbered steps).

## Data Flow
ASCII diagram: upstream cache → processing → downstream cache.

## Downstream Consumers
Who reads this step's output and what they need.
```

---

## PROCESSING.md — Runtime behavior

The "how it runs". Operating principles, phase-by-phase detail, rate limiting mechanics, known gaps.

```
# Step N — Processing

Source of truth: the code in `<main_file>.py`.
Last verified against code: YYYY-MM-DD

## Contract
Numbered principles for how processing works in this step.
(e.g., rate-limiting stack, cold-start strategy, timeout policy)

## Processing
### Overview
Input, output, model, provider, dispatch pattern.

### Processing Strategy
Phase-by-phase breakdown with detail on each.

### Pre-processing Filters
(if applicable) What gets filtered before LLM dispatch.

## Rate-Limiting Machinery
Per-request flow (code snippet), layer-by-layer detail.

## Divergent Paths
Provider differences, bootstrap vs main processing.

## Known Issues and Divergences
Numbered list of gaps vs other steps or ideal behavior.

## Configuration Reference
Tables of runtime parameters and their sources.
```

---

## Naming and heading conventions

- Titles: `# Step N — Name` (em dash, not colon)
- Sections: standard `##` markdown headings (no roman numerals, no letter prefixes)
- Step references: "step 0", "step 1", etc. (lowercase)
- Config references: backtick the field name, e.g., `force_recalc`
