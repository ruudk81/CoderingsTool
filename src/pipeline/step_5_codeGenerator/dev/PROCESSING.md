# Step 5 — Processing

Source of truth: the code in `run_codebook.py` (orchestration),
`consolidation.py` and `codebook_writer.py` (the two LLM calls), and
`grouping.py` (the deterministic middle).

## Contract

Principles for how processing works in this step. Updating code = updating this doc.

### 1. Goal

Build a codebook from taxonomy attributes with the fewest LLM calls that still
produce a defensible result: one call to decide which attributes form one code,
one to write the fixed set of codes. Everything that must HOLD — a whole
partition, pure valence poles, nothing below the threshold standing alone, 100%
coverage — is enforced in Python afterwards, never asked for in a prompt.

### 2. SmoothRequester for every LLM call

Every phase — consolidation, writer (and post-mortem when enabled) —
dispatches through its own `SmoothRequester` instance with
`prepare_fn`/`parse_fn`/`fallback_fn`. No phase shares a rate-limit probe with
another (`known_limits`/`has_server_headers` default `None` throughout); each
fetches its own.

### 3. Retry via SmoothRequester

`max_retries: 2` in every `prepare_fn`. SmoothRequester handles the retry with
backoff. Failure handling differs by phase — see §4.

### 4. Two failure contracts

- **Hard stop**: `resolve_consolidation` raises `RuntimeError` on a failed call.
  Without grouping there is no codebook, so this one phase has no fallback — and
  a silent empty result would reach `check_degeneration` looking like a
  degenerate proposal, which is a different problem with a different fix.
- **Soft failure, degrade gracefully**: `write_codebook` failing to return a key
  means that shape gets deterministic fallback text (`_fallback_text`) instead of
  losing the code; `resolve_postmortem` failing returns no verdicts and the
  codebook stays as it was — adjustment may fail without taking the run with it.

### 5. Model-tier-aware output handling

Instructor + Pydantic validation throughout (Pattern B). Every response model
that lists items back to the model (attribute tags, code keys) is a dynamically
constructed model with a `Literal` enum
constraining the field to exactly what was shown in the prompt — the model
cannot invent, omit within an "exactly one entry per X" contract, or reference
something not in the list.

### 6. Documentation tracks implementation

This PROCESSING.md reflects what the code does now, not what we plan to do.
Known gaps go in `WORK.md`.

### 7. Development code stays clean

No legacy references, no backward-compatibility shims, no dead or redundant code.

## Processing

### Overview

- **Input**: taxonomy attributes + classified ideas (from step 4)
- **Output**: `List[ConsolidatedCode]`, cached as `CodingResultsCache.raw_codes`
- **Models**: `CodebookConfig.model_relations` (consolidation) / `model_writer`
- **Provider**: OpenAI or Azure, via `utils/llm.py` + `SmoothRequester`
- **Dispatch**: `SmoothRequester.process_all()` with
  `prepare_fn`/`parse_fn`/`fallback_fn` callbacks, one instance per phase
- **Call count**: 2 per run — the whole chain is one judgement call and one
  writing call, with Python in between and around

### Processing Strategy

1. **Concept inventory + attribute cards** (no LLM) — flatten step 4's
   classified ideas and taxonomy attributes into `Concept`s: one per attribute
   with ≥1 idea, carrying `frozenset`s of respondent ids per valence pole.
   `build_cards()` then turns each concept into what the model sees: name,
   definition, domain, facet, respondent count, and the most frequent literal
   answers behind it.

2. **Consolidation** (1 LLM call) — `resolve_consolidation()`: one call across
   the whole card inventory. The prompt states the research question, the
   number of respondents, and that each code becomes a row in a report table
   with a percentage next to it. The model returns proposed codes, each a list
   of attribute tags, plus a name and an explanation per code (both advisory —
   the real name comes from the writer). `SmoothRequester(num_tasks=1)`; raises
   `RuntimeError` on failure (hard stop — a silent empty result would be
   indistinguishable from a degenerate proposal).

3. **Grouping** (no LLM) — three deterministic steps on what came back:
   - `repair_partition()` — every attribute placed exactly once. An attribute
     the model forgot becomes its own group; one it placed twice goes to
     whichever group covers the most respondents (union of its members, never
     a sum; ties by member count, then code name); one it invented is dropped.
     Every repair is logged and surfaces in `report_codebook_build`.
   - `build_shapes()` — each group split into positive / negative / neutral
     `CodeShape`s per pole that clears `t_keep` on its own respondent set. A
     pole below the threshold is dropped, never merged into a surviving pole —
     its respondents are counted as `direction_loss` and are represented by no
     code. Clears no pole the threshold, then the group's attributes go to
     Overig.
   - `check_degeneration()` — `n_groups / n_attributes` outside
     `[DEGENERATION_FLOOR, DEGENERATION_CEILING]` is a hard FAIL: reported, and
     the cache write is skipped.

4. **Write codebook** (1 LLM call) — `write_codebook()` with
   `build_writer_prompt`: one call across ALL fixed `CodeShape`s. Each
   shape's direction and member attributes (name + definition, never a count)
   are shown; the model writes name/definition/diagnostic_test/indicators/
   boundary_note per shape, with direction required to be readable in BOTH the
   name and the definition. It may veto a POOLED shape (`nameable: false`) if
   its members share nothing nameable — the veto is ignored on solo shapes
   (single attributes are nameable by definition). A shape the model omits, or
   a failed call, gets deterministic fallback text — the shape itself is never
   dropped by a parsing failure.

5. **Three deterministic guards** (no LLM) — over the FULL reassembled book, in
   order: `resolve_duplicate_names` (same `code_name` twice → keep the larger,
   rename the smaller to its shape's umbrella term), `find_duplicate_definitions`
   (byte-for-byte-normalized duplicate definition text), `find_naming_mismatches`
   (does the written name share a meaningful word with any of its own member
   attribute names?).

6. **Overig sweep + verification** (no LLM, `codebook_io.py` +
   `codebook_verifier.py`) — route every referenced-but-uncovered attribute
   (taxonomy ∪ idea-assigned, by name) into one catch-all code, always emitted.
   Then `build_scorecard()` produces a hard PASS/FAIL plus advisory warnings
   (under-split, mini codes, overlap classes).

### Off by default: stability + post-mortem

With `stability_runs=N` (N ≥ 2), step 2 runs N times instead of once. The first
run becomes the codebook; `measure_stability()` counts, per ATTRIBUTE PAIR, in
how many runs the two sat in one group. Pairs that were neither always together
nor always apart mark where the model has no settled judgement.

`select_candidates()` then picks groups worth a second look: those covering more
than `SHARE_THRESHOLD` of the sample, or containing a pair that wobbled WITHIN
the group. (Not "contains an attribute that wobbles somewhere" — on an inventory
with much movement that is nearly every group, and the post-mortem would reopen
the whole codebook.) `resolve_postmortem()` asks, per candidate, whether it is
one thing or several; `apply_splits()` rejects any verdict that does not exactly
redistribute the group's members.

This path is **off**. Two live runs split 9 of 10 candidates into single
attributes — the null answer. The degeneration gate caught both and wrote
nothing. See WORK.md.

## Rate-Limiting Machinery

Standard `SmoothRequester` stack, no step-5-specific behavior:
concurrency control, RPM/TPM pacing, dispatch staggering, adaptive timeouts
from `perf_model.predict()`, retry with backoff, stats persistence via
`perf_model.observe()`/`save()`.

`data/perf_model.json` keys every ring buffer `(provider:model, phase)`,
where `phase` is the `step5_*` `phase_key` string each `SmoothRequester`
instance passes (see ARCHITECTURE.md's Concurrency & Rate Limiting for the
values). **Step 5 never warms its own perf-model buffers.**
`perfModel.MIN_PHASE_N = 5` — a phase needs five observations before its own
ring buffer is trusted over the pool/cold-default waterfall — and one run of
this chain supplies exactly one observation for `step5_consolidation` and one
for `step5_writer`. Five runs on the same phase key are needed before either
buffer counts, so in practice both phases always start from the pool. This is a
property of a two-call chain, not a gap to fix.

## Divergent Paths

None specific to step 5 — OpenAI and Azure both go through
`SmoothRequester` + `utils/llm.py`.

## Configuration Reference

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| Temperature (consolidation) | `0.0` | `CodebookConfig.temperature_relations` |
| Temperature (writer) | `0.3` | `CodebookConfig` |
| `t_keep_share` / `t_keep_min_respondents` | `0.01` / `3` | `CodebookConfig` |
| `DEGENERATION_FLOOR` / `DEGENERATION_CEILING` | `0.05` / `0.90` | `grouping.py` |
| `SHARE_THRESHOLD` (post-mortem, off) | `0.20` | `postmortem.py` |
| `DEFAULT_RUNS` (stability, off) | `5` | `stability.py` |

The `mece_*` knobs in `CodebookConfig` serve `_quarantine_v1/` only.

### Model-tier keys (config.py `STEP_MODEL` / `STEP_EFFORT`)

`codegen_relations` (consolidation, and the post-mortem when enabled) and
`codegen_writer` — both `("5.4", 5)` (gpt-5.4) / `"high"` reasoning effort as of
this writing. The three `codegen_umbrella_merge` / `codegen_mece_*` keys still exist
because `_quarantine_v1/` reads them. These select the *model*; they are a
different identifier from `SmoothRequester`'s `phase_key` (`step5_*`, see
Rate-Limiting Machinery above), which paces requests and keys the perf model.

### Shared infrastructure

`SmoothRequester` from `utils/smoothRequester.py`. No embedding or clustering
infrastructure — this chain uses neither.
