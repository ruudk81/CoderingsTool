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
  SETS not counts, and `is_drain` inherited from its ref), `t_keep()` (the
  prevalence threshold)
- `attribute_cards.py` — `AttributeCard`: what the model sees per attribute —
  name, definition, domain, facet, respondent count, literal top answers.
  `.tag` is `[A17] Prijs`, the id-plus-name form used across step 5.
  `build_cards(exclude_drains=True)` skips step 4's catch-alls — see Gotchas
- `prompts_consolidation.py` + `consolidation.py` — phase 1, the one
  judgement call: which attributes belong in one code, given the research
  question, the counts and what respondents actually said. Failure is a hard
  `RuntimeError`, never a silent fallback
- `grouping.py` — phase 2/3, **deterministic, no LLM**: `valence_poles` (the
  pole computation two phases share, so they cannot drift),
  `pool_thin_within_facet` (thin facet-mates become one code — only
  sub-threshold material, never across a facet boundary, no constant of its
  own), `repair_partition`
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
- `consensus/` — the consensus candidate: a full second chain beside this one,
  tracked in git next to `_quarantine_v1/` (not `git add -f`'d under `dev/`
  any more, since 2026-08-21). It owns `config_consensus.py`,
  `prompts_consolidation.py`, `consolidation.py` (N runs in one
  `SmoothRequester`, `phase_key` `step5c_consolidation`), `consensus.py`,
  `analysis.py`, `storage.py`, `run_codebook.py`, `view_prompts.py`,
  `view_codebook.py` and its own tests; it borrows everything from
  `build_shapes` onward via relative imports (`taxonomy_input.py`,
  `concept_inventory.py`, `attribute_cards.py`, `code_shape.py`,
  `grouping.py`, `codebook_writer.py`, `prompts_writer.py`,
  `prompts_common.py`, `codebook_io.py`, `codebook_verifier.py`). Since
  2026-08-21 it has one clickable runner: `consensus/run_codebook.py` carries
  a settings block at the top (`ACTIE`, `CONFIG`, `RUNS`, `TAU`, `SET`, ...)
  and dispatches on click to one of five actions — `alles`, `verzamelen`,
  `codeboek`, `analyse`, `vergelijk`. `codeboek` (and `alles`, which ends
  with one) produces the same five deliverables production does: the
  codebook cached under `mece_codes` — the SAME cache key as production, so
  steps 6 and 7 can run on consensus codes, which means the cache holds one
  codebook at a time and the last chain to run wins — cost under its own
  step `step_5_consensus`, prompt export under doctype `prompts_step5c`,
  perf stats under `phase_key` `step5c_consolidation`, and `log_step5c.txt`.
  That log is written on any click. `verzamelen` alone — the action that makes
  the RUNS consolidation calls, the expensive side of the chain — records its
  own cost (`step_5_consensus`, phase `"consolidation"`) and its own prompt
  export (`prompts_step5c`) too, plus the perf stats and its partitions file;
  it does not touch `mece_codes`, that write stays exclusive to `codeboek`/
  `alles`. `codeboek` also refuses to run when the set's attribute universe no
  longer matches the current step-4 cache — the same guard `vergelijk` uses
  between two sets, applied here between one set and the live cache, since
  `codeboek` is the action that writes the shared cache. `analyse`/`vergelijk`
  stay read-only (no LLM calls, no deliverable beyond the log). Not promoted:
  the consensus measurement itself missed its own bar (ARI 0.788 against a
  required 0.90) — see WORK.md
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
| `codegen_relations` | `step5c_consolidation` | `consensus/consolidation.py` — its own key because its VORM differs (N parallel tasks, not one); its writer call keeps `step5_writer`, see the `consensus/` bullet above |

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
- **Step 4's catch-alls can be kept off the cards — opt-in, off by default.**
  Step 4 builds an `other` attribute under every facet and marks it with
  `drain_key`; `AttributeRef` and `Concept` carry that as `is_drain`, and
  `build_cards(exclude_drains=True)` skips it. `run_codebook.py` does NOT pass
  the flag: production still shows catch-alls to the model, pending the
  promotion decision on the consensus experiment. Recognition is
  on the key and never on the name — the name is in the survey language and may
  be rewritten (step 4's `drains.py` states the same rule). **The reason is
  NOT that a catch-all has no subject** — that motivation was refuted on
  2026-08-21 and removed here. A catch-all is facet-scoped: "the rest within
  Political direction" has a subject, merely an unspecified one, and putting it
  with its facet's main code is defensible. The reason is what such a merge does
  to a MEASUREMENT: on the ASN set (2026-08-20) each catch-all merged with its
  namesake attribute at 28-29 of 30 runs — a facet-and-name match, so near
  automatic — and therefore topped the co-association matrix over buckets holding
  one or two respondents. That is recurrence without prevalence. The Concepts
  stay (their respondents belong in the bookkeeping) and `apply_overig_sweep`
  covers them, but note the price: that sweep emits ONE global `Overig`, so with
  the flag on those respondents lose their facet context. A third way — off the
  cards, but routed deterministically to their own facet's code, as
  `pool_thin_within_facet` already does — is open in WORK.md.
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
