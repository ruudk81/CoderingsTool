# Step 5 — Code Generator

## Purpose
Builds a MECE codebook from step 4's taxonomy attributes. One LLM call decides
which attributes form one code, Python splits every group into pure valence
poles and guards the partition, a second LLM call writes the text. No
embeddings, no per-domain generation, no theme-count clustering.

This chain replaced the v1 chain on 2026-08-18 (see
`.superpowers/specs/2026-08-18-step5-v2-promotienotitie.md`). v1 — relations,
deterministic consolidator, iterative MECE rounds — is retired in
`_quarantine_v1/`; it still imports and its tests still run, so it can be
measured against if this chain fails on another dataset.

It lived in a `v2/` subpackage until 2026-08-19; with no v1 beside it, the
version label had stopped naming anything and the modules moved up into the
step folder. **`v2` is not a path, a module or an identifier in step 5 any
more** — a leftover reference to one is stale, not a second chain.

## Key Files
- `taxonomy_input.py` — the only place that knows step 4's shape (`IdeaUnit`,
  `AttributeRef`, `build_idea_units`, `build_attribute_refs`); step 4 can change
  underneath this without the rest of step 5 noticing
- `concept_inventory.py` — `Concept` (one per attribute with ≥1 idea, respondent
  SETS not counts), `t_keep()` (the prevalence threshold)
- `attribute_cards.py` — `AttributeCard`: what the model sees per attribute —
  name, definition, domain, facet, respondent count, literal top answers.
  `.tag` is `[A17] Prijs`, the id-plus-name form used across step 5
- `prompts_consolidation.py` + `consolidation.py` — phase 1, the one
  judgement call: which attributes belong in one code, given the research
  question, the counts and what respondents actually said. Failure is a hard
  `RuntimeError`, never a silent fallback
- `grouping.py` — phase 2/3, **deterministic, no LLM**: `repair_partition`
  (every attribute in exactly one group, whatever the model returned),
  `build_shapes` (each group split into pure valence poles → `CodeShape`),
  `check_degeneration` (a proposal that groups nothing, or everything, is
  reported and blocks the cache write)
- `code_shape.py` — `CodeShape` plus `_shape_lookup`/`_match_shape`: the form of
  a code, and how a written code is matched back to the form it came from.
  `stored_valence()` also translates `two_pole`'s fourth valence value
  (`non_negative`) to `neutral` for `models.py`'s three-value contract —
  experiment-only today, see ARCHITECTURE.md
- `prompts_writer.py` + `codebook_writer.py` — phase 4: one LLM call
  writes name/definition/diagnostic_test/indicators/boundary_note for every
  fixed `CodeShape`; also the three deterministic guards over the full
  reassembled book (`resolve_duplicate_names`, `find_duplicate_definitions`,
  `find_naming_mismatches`). `prompts_writer.py` holds the plumbing BOTH chains
  share (`_code_block`, `_ordered`, `make_writer_model`, `CodeText`,
  `WriterResult`) plus the production `build_writer_prompt`; v1's own variant
  sits in `_quarantine_v1/prompts_writer_v1.py`
- `prompts_common.py` — `INSTRUCTOR_HINT` and `_shuffled`, what every prompt
  module in step 5 needs
- `codebook_io.py` — everything around the chain: the three loaders, the cache
  write, `apply_overig_sweep` (the 100%-coverage guarantee), `run_scorecard`,
  `print_codebook_results`, `save_prompts_to_json`
- `codebook_verifier.py` — deterministic post-generation scorecard
  (`build_scorecard`, `format_scorecard`) — PASS/FAIL + advisory warnings
- `config_codeGenerator.py` — `CodebookConfig` dataclass
- `run_codebook.py` — step runner. `generate_codebook()` is the
  reusable orchestration entry point; `run_codebook()` is the production
  entry point built on top of it
- `stability.py` — measures how firmly phase 1 lies by repeating it: per
  ATTRIBUTE PAIR, in how many runs did the two sit together. Deliberately does
  **not** derive a consensus partition (see the module docstring)
- `postmortem.py` + `prompts_postmortem.py` — post-generation splitter.
  **Off** (`stability_runs=0`); its question form produces the null answer, see
  WORK.md
- `view_codebook.py` — read-only: the final cached codebook
- `view_prompts.py` — read-only: the two live phases' prompts as captured by
  `PromptPrinter`, with instructor's tool definition (thin wrapper around
  `utils/promptViewer.py`); reconstructs each call's enum-constrained runtime
  response model from identifiers stored in that capture's own metadata,
  falling back to the base model only when that metadata is absent
- `_quarantine_v1/` — the retired v1 chain; nothing in it runs in production.
  Its `prompts_writer_v1.py` is the one thing it does NOT share with the live
  chain, and `run_codeGenerator.py` passes it explicitly to `write_codebook`
- `dev/experiment_consensus/` — consensus-over-N-runs experiment
  (`consensus.py`, `analysis.py`, `storage.py`, `stability_bridge.py`,
  `run_experiment.py`). Read-only against the pipeline cache — writes nothing
  under `mece_codes`. Not run beyond smoke tests. Design:
  `.superpowers/specs/2026-08-20-step5-consensus-experiment-design.md`
- `CodingResultsCache` (cache model) lives in `src/models.py` — the single
  source of truth for all cross-step models
- `ConsolidatedCode` lives in `src/models.py` alongside `CodingResultsCache` —
  it is the codebook entry as cached and read by steps 6 and 7, not a response
  model (the three instructor `response_model` slots use `make_writer_model`,
  `make_consolidation_model` and `make_postmortem_model`). Moved there
  2026-08-18; `prompts_codeGenerator.py` held nothing else and is gone

## Input / Output Contract
- **Input**: `TaxonomyResultsCache` from step name `taxonomy` (prefix 005) +
  `ExtractionMetadata` (`extracted_ideas`, prefix 004) + `TaxonomyClassifiedModel`
  (growing model `taxonomy_classified` from step 4, for the literal answers on
  each attribute card)
- **Output**: `CodingResultsCache` cached under step name `mece_codes` (prefix 006)
  - `raw_codes`: list of `ConsolidatedCode` dicts (code_name, definition,
    diagnostic_test, valence, typical_indicators, source_attributes,
    source_attribute_ids, code_id)
  - `codebook_narrative`: unused by this chain (legacy P8/P9 field, kept on the
    model for old caches; always `""` on a codebook this chain writes)
  - Does **not** contain idea embeddings — this chain never computes any
- **Cache check**: `run_codebook(force_recalc)` honors `force_recalc`; skips
  generation when `is_metadata_cache_valid(FILENAME, "mece_codes", variable_key)`

## LLM Usage
Two phases, resolved via `get_step_model()`: `codegen_relations` (consolidation
reuses that rung and its config knobs) and `codegen_writer`. Both on `("5.4", 5)`
(gpt-5.4) at `"high"` reasoning effort (`STEP_EFFORT`). Call count: **2** — one
consolidation, one writer. With `stability_runs=N` the consolidation call runs
N times and the post-mortem adds one, but that path is off.

Dispatch: one `SmoothRequester` instance per call, constructed inside the module
that owns the call — `run_codebook.py` itself never constructs one, it only
orchestrates. The `get_step_model()` keys select the *model*; `SmoothRequester`'s
own `phase_key` (a separate identifier, used for rate-limit pacing and the perf
model, see PROCESSING.md) is a `step5_*` string per call site:

| `get_step_model()` key | `phase_key` | Module |
|---|---|---|
| `codegen_relations` | `step5_consolidation` | `consolidation.py` |
| `codegen_writer` | `step5_writer` | `codebook_writer.py` |
| `codegen_relations` | `step5_postmortem` | `postmortem.py` (off) |

## Shared Utils
- `utils/smoothRequester.py` — `SmoothRequester` for every LLM call
- `utils/llm.py` — `token_tracker`
- `utils/cacheManager.py` — cache load/save
- `utils/identity.py` — `ensure_codebook_ids` (mints `K#`, resolves
  `source_attribute_ids`)
- `utils/costTracker.py` — one `record_phase` call wraps the whole chain
- `utils/promptPrinter.py` — prompt capture; both phases accept a
  `prompt_printer` and forward it (`run_codebook()` → `generate_codebook()`
  → `resolve_consolidation`/`write_codebook`)
- `utils/exportNaming.py` — `export_filename()` names the captured-prompts
  JSON the same way every other step does
- `utils/saveVerbose.py` — `VerboseCapture`, wrapping the `__main__` block
  only (see Gotchas)

## Gotchas
- **`taxonomy_input.py` uses plain attribute access, never `getattr(..., default)`
  — deliberately.** An empty field is a valid value (`or ""` keeps it); a field
  that does not EXIST is a broken contract with step 4 and must raise. A default
  collapses those two cases into one, and the second silently became the first:
  a renamed `valence` turned every idea neutral, after which the direction split
  disappeared without a single error. Same reason `build_attribute_refs` reads
  `attribute["attribute_name"]` by key. The two `attribute_definition` /
  `attribute_description` reads stay `.get` — that one IS a genuine either/or.
- **Valence purity is by construction, not by instruction.** `build_shapes`
  splits every group into separate positive / negative / neutral codes from the
  respondent sets; the writer is never asked to keep a code single-signed. A
  prompt rule would be advisory — this cannot produce a mixed code at all.
- **Respondent sets are unioned, never summed.** A respondent who answered on
  two attributes of the same group counts once. `weight()` in `grouping.py` and
  `group_respondents()` in `postmortem.py` both take a union; summing `n_resp`
  double-counts and was a real bug in both, caught in review.
- **Concepts only exist for attributes with ≥1 classified idea.** An attribute
  with zero ideas never becomes a `Concept` and is invisible to the chain. The
  actual 100%-coverage guarantee is `apply_overig_sweep` in `codebook_io.py`,
  which sweeps ALL taxonomy attributes by name regardless of whether they ever
  entered the chain.
- **Two different shapes can get the same written `code_name`.** The writer is
  free text; nothing stops it from naming two unrelated `CodeShape`s the same
  thing in one batch call. `resolve_duplicate_names` (codebook_writer.py) is
  the deterministic backstop, run over the FULL reassembled book — never trust
  `taken_names` alone (a prompt-level ask, not a guarantee).
- **Reassembly after `write_codebook` must key on the shape, never on the
  written name.** A dict comprehension keyed on `code_name` collapses the moment
  two shapes share a name — a real, shipped bug on an ASN run. `_match_shape`
  (`code_shape.py`) matches on (source-attribute names, valence), the two things
  the writer echoes back.
- **Degeneration blocks the cache write, and only the cache write.** A proposal
  that groups nothing (or everything) is reported, the codebook and scorecard
  still print, but nothing lands under `mece_codes` — step 6 must never read a
  degenerate book silently. This gate fired twice on real post-mortem runs.
- **`write_codebook` may veto a shape**, so `codes` can be shorter than `shapes`.
  Anything zipping the two lists breaks; match through `_shape_lookup` instead.
- **Prompt capture is wired into `run_codebook()` itself, not only
  `__main__`.** It constructs the `PromptPrinter`, threads it through both
  phases and calls `save_prompts_to_json()` at the end — so a call from
  `run_pipeline.py` or the app captures prompts under the canonical
  `exportNaming.export_filename` name exactly like a direct run does. This is
  the opposite of the verbose-log gotcha below — read both together.
- **`VerboseCapture` wraps only the `__main__` block**, not `run_codebook()`
  itself — a direct run writes `log_step5.txt` to `exports/verbose_logs/`, a
  call from `run_pipeline.py` or the app does not. Steps 3 and 4 do exactly the
  same; this is the project convention, not a step-5 gap.
- **Cost ledger is one phase, not two.** `run_codebook()` wraps the whole
  `generate_codebook()` call in one `token_tracker` snapshot pair and records
  it as a single `"codebook_generation"` phase under step `step_5_code_generator`
  (`exports/costs/*_kosten.json`) — both phases resolve to the same model rung,
  so a finer split buys nothing. A deliberate choice, not an omission.
- **`write_codebook`'s `prompt_builder` default is the PRODUCTION prompt**, and
  the retired v1 chain is the one that passes its own (`build_writer_prompt_v1`).
  It used to be the other way round, which meant a caller who forgot the
  argument silently got the prompt of a chain that no longer runs. The two
  prompts differ by one rule — the production one requires the direction to be
  readable in both the code's name and its definition. They must stay different:
  that v1's lacks it is a measured property of that chain (19 direction-less
  names out of 42 codes, 2026-08-14), and levelling them destroys the baseline.
- **The post-mortem splitter is off and must stay off** until its prompt is
  revised. `stability_runs` defaults to 0 everywhere, including `__main__`. Two
  live runs split 9 of 10 candidate groups into single attributes; the
  degeneration gate caught both. See WORK.md.

## Processing Phases
1. **taxonomy_input + concept_inventory + attribute_cards** — flatten step 4's
   classified ideas and taxonomy attributes into `Concept`s (respondent sets,
   never counts), then into the cards the model sees
2. **consolidation** — one LLM call over the whole inventory: which attributes
   form one code, given the research question, the counts and the literal answers
3. **grouping** (deterministic) — repair the partition, split each group into
   pure valence poles, check for degeneration
4. **codebook_writer** — one LLM call writes every fixed `CodeShape`'s text
5. **three deterministic guards** — duplicate names, duplicate definitions,
   naming mismatch, over the full reassembled book
6. **Overig sweep + verification** — route unplaced attributes into a
   catch-all, then `build_scorecard()` for a PASS/FAIL readout

## Dev Docs
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [CACHE_LOGIC.md](CACHE_LOGIC.md) — caching contracts
- [PROCESSING.md](PROCESSING.md) — processing flow
- [WORK.md](WORK.md) — known gaps and planned fixes
