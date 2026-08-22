# Step 5 — Architecture

## Design Intent

A codebook is built from taxonomy attributes in four stages: decide which
attributes form one code, split each group into pure direction poles, guard the
partition, write the text.

The division of labour is the whole design: **the model supplies judgement,
Python supplies guarantees.** Everything that must hold — every attribute placed
exactly once, no code mixing a positive and a negative pole, nothing below the
prevalence threshold standing alone, 100% coverage — is enforced in Python after
the model has answered. Nothing that must hold is asked for in a prompt.

Key design choices:

- **The model is told what the codebook is FOR.** The consolidation prompt names
  the research question, says each code becomes a row in a report table with a
  percentage next to it, and shows what respondents literally said. v1's grouping
  phase deliberately withheld counts and context to avoid bias, and got grouping
  on form — name similarity, level — instead of on purpose. Withholding the goal
  did not remove the bias; it removed the basis for judgement.
- **Respondent sets, not counts, everywhere internal.** `Concept` and `CodeShape`
  carry `frozenset`s of respondent ids; combining two groups is a set union,
  never a sum — a respondent who shows up in two source attributes is not
  double-counted. This has been got wrong twice in review (`weight()` in
  `grouping.py`, `group_respondents()` in `postmortem.py`), both times by summing
  `n_resp`.
- **Valence purity is a construction, not an instruction.** `build_shapes` splits
  every group into separate positive / negative / neutral codes from the
  respondent sets. A code IS a pole. The writer is never asked to keep a code
  single-signed, because a prompt rule would be advisory; this cannot produce a
  mixed code at all. The cost is `direction_loss`: a pole too small to clear the
  threshold has nowhere to go, and the scorecard's `under_split_codes` is what
  makes that visible.
- **The form is fixed before the text is written.** `grouping.py` decides how
  many codes there are, which attributes each contains, and its direction, with
  no LLM call. `codebook_writer.py` only fills in name / definition / diagnostic /
  indicators for a `CodeShape` that already exists — it cannot add, remove, or
  resize a code.
- **A returned partition is repaired, never trusted.** `repair_partition` takes
  whatever the model returned and makes it a partition: a forgotten attribute
  becomes its own group, a duplicated one goes to whichever group covers the most
  respondents (union of its members, never a sum), an invented one is dropped. Every repair is logged and reported. The enum on the response
  model bounds the vocabulary; it cannot bound completeness.
- **Degeneration is reported and blocks the write.** A proposal that groups
  nothing (as many groups as attributes) or everything (one group) is not a
  codebook. `check_degeneration` reports it and `run_codebook` skips the cache
  write — the codebook and scorecard still print, because the point is to show
  what went wrong, not to hide it. This gate has fired twice on real runs.
- **Naming collisions are caught after the fact, never trusted to a prompt.** The
  writer is asked to avoid names already taken (`taken_names`), but three
  deterministic checks run over the fully reassembled book regardless of what the
  model claimed: duplicate names, duplicate definitions, and a naming mismatch
  (does the written name share a meaningful word with any of its own member
  attributes?).
- **A test whose passing condition the model controls is not a test.** This is the
  pattern the whole chain is shaped around, and it keeps recurring. A partitioning
  question ("group these names") returned the trivial answer on a live run — 45
  names in, 45 groups out. A MECE probe that asked the model to write its own
  separation rule and then judge against it never merged anything: a model asked
  to write a rule always writes one it can satisfy. Most recently, the
  post-mortem asked "is this group one thing, or several?" and split 9 of 10
  candidate groups into single attributes — the null answer again, now in a new
  costume. The lesson generalises: an open question about structure has a cheap
  correct-looking answer, and the model will find it. Forced lookups ("which two
  of these belong together LEAST?") do not have that escape hatch.

## Pipeline Overview

```
Input: TaxonomyResultsCache (step 4, prefix 005)
       + ExtractionMetadata (step 3, prefix 004)
       + TaxonomyClassifiedModel (step 4, prefix 005, growing model)

STAGE 1 — taxonomy_input + concept_inventory + attribute_cards (no LLM):
  classified ideas -> IdeaUnit (flat, per idea)
  taxonomy attributes -> AttributeRef (flat, per attribute)
  -> Concept per attribute with ≥1 idea (respondent SETS, not counts)
  -> with exclude_drains: step 4's catch-alls keep their Concept but get no
     card (off by default; the experiment turns it on)
  -> t_keep(n_resp_total) = max(t_keep_min_respondents, round(t_keep_share * n))
  -> AttributeCard per concept: name, definition, domain, facet, n_resp,
     top literal answers

STAGE 2 — consolidation (1 LLM call):
  the whole card inventory + the research question + n_respondents
  -> ConsolidationResult: proposed codes, each a list of attribute tags
  (failure raises — without grouping there is no codebook)

STAGE 3 — grouping (deterministic, no LLM):
  repair_partition: every attribute in exactly one group; forgotten -> own
                    group, duplicated -> group with the most respondents,
                    invented -> dropped
  build_shapes:     each group split into positive/negative/neutral poles that
                    clear t_keep; a pole below it is dropped (direction_loss)
  check_degeneration: n_groups/n_attributes outside [0.05, 0.90] -> FAIL

STAGE 4 — codebook_writer (1 LLM call):
  CodeShapes + their member Concepts -> name/definition/diagnostic_test/
  indicators/boundary_note per shape -> List[ConsolidatedCode]
  (a `nameable: false` verdict on a POOLED shape drops it — recorded as a
  VETO; the same verdict on a solo shape is ignored)

DETERMINISTIC GUARDS (over the full reassembled book):
  resolve_duplicate_names -> find_duplicate_definitions -> find_naming_mismatches

Overig sweep (codebook_io.py, deterministic):
  referenced = taxonomy attributes ∪ idea-assigned attributes (by name)
  orphans = referenced - (union of every code's source_attributes)
  -> one catch-all ConsolidatedCode, always emitted

VERIFICATION (codebook_verifier.py, deterministic):
  build_scorecard() -> PASS/FAIL + advisory warnings -> printed to console

Output: CodingResultsCache (partition_set, partition_results, raw_codes)
        Saved as step="mece_codes", prefix 006 — unless degeneration fired
```

### Off by default: stability + post-mortem

`stability_runs=N` (N ≥ 2) repeats stage 2 N times, measures per ATTRIBUTE PAIR
in how many runs the two sat together (`stability.py`), and feeds the
wobbling pairs to a post-generation splitter (`postmortem.py`). The first run
becomes the codebook; the rest only direct attention.

The machinery is built and tested but **off** (`stability_runs=0` everywhere).
Its question form produces the null answer — see the last Design Intent bullet
and WORK.md. The stability measurement on its own is usable and costs one extra
call per run.

### Two-pole valence (experiment support)

`build_shapes(two_pole=True)` (`grouping.py`) replaces the three-way positive /
negative / neutral split with two poles: `non_negative` (positive ∪ neutral)
and `negative`. Since 2026-08-22 `ConsolidatedCode.valence` (`models.py`) is a
FOUR-value `Literal` that accepts `non_negative` directly, and
`code_shape.stored_valence()` translates nothing.

It translated `non_negative` to `neutral` until then, and that was not a
rounding but a loss of meaning: `neutral` means "descriptive, no direction"
while `non_negative` means "explicitly not a complaint". Step 6's `opposes()`
deliberately leaves `neutral` out of its opposites table — descriptive material
has no counter-pole — so the direction guard never fired on a codebook built
with two poles. That was the path being run daily. See WORK.md.

`stored_valence()` still exists and is still called on both sides of the
shape↔code match, so the two sides cannot drift apart if a translation is ever
reintroduced.

This exists for `consensus/` (see CLAUDE.md's Key Files and "The consensus
candidate chain" below). `run_codebook()` never passes `two_pole=True`, so the
production chain still runs three poles — this is live code in `grouping.py`,
`code_shape.py`, `codebook_writer.py` and `prompts_writer.py`, not a dead
branch, but it has no production caller today — the candidate chain in
`consensus/` does, and it writes under the same `mece_codes` key.

## Prompt Builders & Response Models

| Stage | Builder | Response Model | Notes |
|-------|---------|-----------------|-------|
| 2 | `build_consolidation_prompt()` | `ConsolidationResult` (dynamic `Literal` on each code's `topics`) | Counts and literal answers ARE shown — the model needs them to judge what a reader would want as one row |
| 4 | `build_writer_prompt()` | `WriterResult` (dynamic `Literal` on `CodeText.key`) | Shape (count, members, direction) is fixed input, never model output. Direction must be readable in BOTH name and definition |
| — | `build_postmortem_prompt()` | `PostMortemResult` (dynamic `Literal` on group label + topic tag) | Off. One part-number per topic, so a forgotten topic is visible instead of nested away |

Every builder ends its prompt with `INSTRUCTOR_HINT` ("provide your output as
valid JSON following the response schema provided") — `Field(description=...)`
alone is not enough; without the literal hint, 23+/56 tasks failed in earlier
testing elsewhere in this codebase.

### Deterministic ordering (`_shuffled`)

Every prompt that lists more than one item orders them via `_shuffled()`
(`prompts_common.py`): sorted by `md5(stable_id)`, never by the caller's
prevalence-sorted order and never by raw id (identity.py mints ids sequentially
per domain, so id order would still leak domain structure as contiguous blocks).
The hash is a pure function of a stable id, so the order is reproducible across
runs while carrying no signal about frequency, domain, or input order.

## Leak Discipline

The rule is not "show the model as little as possible" — it is **show what
supports the judgement, withhold what would answer it for the model.**

- Stage 2 (consolidation) sees name, definition, domain, facet, respondent count
  and literal answers. Step 4's catch-alls are residue by construction and carry
  no groupable meaning; `exclude_drains` keeps them off the cards, but it is off
  in production until the consensus experiment is decided. Counts are shown deliberately: a code's size is part of
  what makes it worth a table row. What it does not see is any suggestion of how
  many codes there should be, and list order carries no signal (`_shuffled`).
- Stage 4 (writer) sees each shape's already-decided direction — a fact to
  respect, not to infer — plus its members' name and definition.

## Concurrency & Rate Limiting

Every LLM call goes through its own `SmoothRequester` instance (`model=`,
`phase_key=`, `prepare_fn`/`parse_fn`/`fallback_fn`), constructed inside the
module that owns the call (`consolidation.py`, `codebook_writer.py`) —
`run_codebook.py` orchestrates the chain but never constructs a
`SmoothRequester` itself. No stage pre-fetches or shares rate limits across calls
(`known_limits`/`has_server_headers` default `None` everywhere) — each phase
probes its own limits independently. `phase_key` is `step5_consolidation` /
`step5_writer` (and `step5_postmortem` when enabled) — a separate identifier
from the `get_step_model()` keys in Configuration below, used for rate pacing and
the perf model (mechanics in PROCESSING.md).

Two failure contracts, by design:
- **Hard stop**: `resolve_consolidation` raises `RuntimeError` if its call fails
  — without grouping there is no codebook, and this chain has no fallback for it.
  A silent empty result would look like a degenerate proposal and be misread.
- **Soft failure**: `write_codebook` returns deterministic fallback text per
  shape; `resolve_postmortem` returns no verdicts and the codebook stays as it
  was — adjustment may fail without taking the run with it.

## Identity Contract (naam-als-identiteit)

Every domain/facet/attribute/code carries an immutable id (`D#/F#/A#/K#`,
`src/utils/identity.py`) minted once at artifact finalization; names are
display-only. `ensure_codebook_ids` runs at cache-save (`cache_mece_results`),
minting `K#` in list order (written codes, then Overig) and resolving
`source_attribute_ids` from `source_attributes` names against the taxonomy's own
id space. Consolidation internals stay name-based — the LLM never sees an id,
only `[A17] name`-tagged references (`AttributeCard.tag`) where disambiguation is
needed. Pre-id artifacts are normalized in memory at load; disk is never mutated
by normalization.

## Configuration

**`CodebookConfig`** (dataclass in `config_codeGenerator.py`):

| Field | Default | Purpose |
|-------|---------|---------|
| `model_relations` | `get_step_model("codegen_relations")` | Consolidation model (and post-mortem, when enabled) — `("5.4", 5)` (gpt-5.4) at `"high"` reasoning effort as of this writing |
| `model_writer` | `get_step_model("codegen_writer")` | Writer model — same rung and effort |
| `temperature_relations` | `0.0` | Set on the dataclass, never sent to the API — see note below |
| `temperature_writer` | `0.3` | Same |
| `max_tokens_relations` / `writer` | `16000` / `16000` | Per-phase token budgets |
| `t_keep_share` / `t_keep_min_respondents` | `0.01` / `3` | Prevalence threshold: `max(min_respondents, round(share * n))` |

Both temperature fields are dead configuration: `utils/llm.py` only adds
`temperature` to the request for non-reasoning models, and both step-5 phases
run on a reasoning model. Neither value reaches the API call — see WORK.md.

The `model_umbrella_merge` / `model_mece_detect` / `model_mece_probe` fields and
the `mece_*` knobs below them serve `_quarantine_v1/` only. They stay because the
quarantined chain still has to import and run.

**Degeneration bounds** (`grouping.py`, not in `CodebookConfig`):
`DEGENERATION_FLOOR = 0.05`, `DEGENERATION_CEILING = 0.90` on
`n_groups / n_attributes`. Reasoned starting values, not measured — revisit once
there are runs on more than one dataset.

## The retired v1 chain

`_quarantine_v1/` holds the chain this replaced on 2026-08-18: per-attribute
`synonym_of` + `umbrella_name` relations (2 LLM calls), a deterministic
consolidator that pooled under the threshold by climbing umbrella → domain, and
iterated MECE rounds (Pass A forced lookup + Pass B blind idea probe, 2 calls per
round). Three to fourteen LLM calls, against the current chain's two.

Everything it needs beyond its own folder is shared with the live chain
(`codebook_io.py`, `codebook_writer.py`, `code_shape.py`, `concept_inventory.py`,
`prompts_common.py`, `taxonomy_input.py`) — with one exception: its writing
prompt, `_quarantine_v1/prompts_writer_v1.py`, which it passes to
`write_codebook` explicitly. That is the only place the two chains still differ
downstream of consolidation, and it is deliberate (see CLAUDE.md's Gotchas).

It is kept, not deleted: the current chain is measured on one dataset. Its
modules still import and its tests still run, so it can be run against the
current chain on a second dataset. Why it
lost, with the numbers: `.superpowers/specs/2026-08-18-step5-v2-promotienotitie.md`.

## The consensus candidate chain

`consensus/` is a second full chain beside this one, not a variant of it.
Where the two differ:

- **Stage 2 runs N times, not once.** `consensus/consolidation.py` builds one
  `SmoothRequester` over N consolidation tasks (`num_tasks=N`, `phase_key`
  `step5c_consolidation`) instead of production's `num_tasks=1`.
- **A deterministic consensus stage sits between consolidation and grouping.**
  `consensus/consensus.py` builds a co-association matrix over the N proposed
  partitions (how often two attributes sat together) and cuts it at a fixed
  `tau` (complete linkage) into one partition. Production has no such stage —
  its single consolidation call's output goes straight to `grouping.py`.
- **`build_shapes` runs with `two_pole=True`.** The candidate splits every
  group into two poles (`negative` / `non_negative`) instead of three
  (positive / negative / neutral) — see "Two-pole valence" above for the
  mechanism both chains share.
- **`exclude_drains=True`.** The candidate keeps step 4's catch-alls off the
  consolidation cards; production shows them (see WORK.md, "Vangnetten op de
  kaarten").
- **A pole that misses `t_keep` is not dropped.** Since 2026-08-22 the
  candidate's `build_shapes` hands every fallen pole to `pool_minority_poles`,
  which unions them per (facet, valence): a union that clears `t_keep` becomes
  a main code, one between `t_keep_min_respondents` and `t_keep` becomes a
  child of Overig (`CodeShape.origin == "child"`, never vetoable), one below
  the floor becomes true-Overig. Poles are collected from EVERY group. The
  narrower rule shipped first — collect only where a sibling pole survived,
  since that is where material is counted under an opposite-facing code — and
  was broadened on 2026-08-22: a group that went to Overig whole left its
  minority material undifferentiated there, and the goal is that such material
  gets a name of its own. Measured on set 7 (luna, tau=0.7, `t_keep` 23):
  32 main codes / 11 children / 113 respondents in children, against 30 / 8 / 74
  under the narrow rule. The old
  `direction_loss` metric counted what fell away and is replaced by
  `coverage_recovered` — the unique respondents who gained a code they would not
  otherwise have had. Production still drops the pole, and its attribute stays a
  source of the surviving, opposite-facing sibling code.
- **Two writing calls, split on origin.** `child` shapes go to
  `write_miscellaneous` (own prompt in `prompts_miscellaneous.py`, own phase
  `step5c_miscellaneous`, no veto and no `nameable` field); everything else goes
  to `write_codebook`. The main-code names travel along as `taken_names`, and
  `resolve_duplicate_names` runs over the REUNITED list. Both result lists are
  matched back to their shapes through ONE `_shape_lookup` over all shapes —
  never zipped, since order is not identity.
- **The hierarchy lives in a field.** `apply_overig_sweep` returns the Overig
  CODE (not its name) and mints the K# there, after which
  `run_codebook.link_children_to_overig` sets each child's `parent_code_id`.
  Ordering is forced: the parent has no id before the sweep. Production's sweep
  returns a name and knows no children.

Where they no longer differ only in phase 1: everything from `build_shapes` to
the Overig sweep diverged on 2026-08-22. `grouping.py`, `code_shape.py`,
`codebook_writer.py`, `codebook_io.py` and `test_grouping.py` stopped being
copies that day and are now consensus's own versions, listed in
`test_zelfstandigheid.py`'s `EIGEN_VERSIE` with the reason per file. What is
still the same code: the three deterministic naming/definition guards, the
scorecard, and everything upstream of grouping. Since 2026-08-22 none of it is shared by
import: each of the eleven modules involved lives as its own copy inside `consensus/` (see
CLAUDE.md's Key Files), so `consensus/` imports nothing from outside itself
within step 5. Three guards in `consensus/test_zelfstandigheid.py` keep that
true — an import guard, a drift guard that fails the moment a copy stops
matching its production original, and a `project_root` guard added after the
2026-08-22 promotion rehearsal. Both chains write `mece_codes`; whichever runs
last is what steps 6 and 7 read.

**Status: not promoted.** The consensus measurement this chain exists to
support missed its own bar — ARI 0.788 between two independent consensus
partitions against a required 0.90 (see WORK.md, "Consensus over N runs").
Becoming a full candidate chain on 2026-08-21 changed the form the experiment
lives in (a package tracked like `_quarantine_v1/` instead of a script under
`git add -f`), not that result.
