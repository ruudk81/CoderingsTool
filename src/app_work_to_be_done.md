# App — Work to be done

Status of the Streamlit UI (`src/app.py` + `src/app_backend.py`). Rebuilt 2026-06-08
("option B": fresh, slim app that lifts the proven cache/verbose/navigation mechanism
from the old `src/app_old.py` and rewrites the rendering for the current 8-step pipeline).

---

## Architecture (how it works)

The **cache is the source of truth**. The UI is a thin orchestrator; it never owns the
canonical results — it probes the cache each rerun.

- **`src/app_backend.py`** — Streamlit-free, importable/testable orchestrator:
  - `DatasetSpec` — run identity (filename, var_name, sample_size) → `variable_key`.
  - `list_cached_datasets()` — resumable datasets, read straight from the `cache_metadata`
    table (no fragile filename parsing).
  - `step_status()` / `is_step_done()` / `max_completed_step()` — live "done" probe per step,
    using the exact same cache checks each runner does before skipping. **No parallel
    `completed_steps` state to drift.**
  - `run_step()` — dispatch a step to its pipeline runner, wrapped in `VerboseCapture`.
    Includes a **safety guard**: after a run, asserts the result exists under THIS dataset's
    keys (loud ⚠️ if a runner wrote to the wrong key).
  - `invalidate_from()` — cascade invalidation (force-recalc from step N onwards).
  - `load_codebook()` / `load_assignments()` / `export_path()` / `find_verbose_log()`.
  - Self-test: `python app_backend.py`.
- **`src/app.py`** — UI only: step wizard (0–7), live status icons, run/re-run, verbose-log
  expander, results from cache + the step-7 Excel export. Bilingual nl/en. Validated with
  `streamlit.testing.v1.AppTest`.

Run: from the project root → `streamlit run src/app.py --server.headless true` (port 8501).

---

## Done

- [x] New `app.py` + `app_backend.py` (option B), `app_old.py` kept as reference.
- [x] Dataset discovery / resume from the cache DB (upload OR pick a cached dataset).
- [x] Step wizard 0–7 with monotonic gating (a step is reachable if done or = next).
- [x] Run / re-run per step; **cascade invalidation** ("Herverwerk vanaf stap N").
- [x] Verbose execution log per step (expander).
- [x] Result views for **step 5** (codebook table), **step 6** (code-frequency table),
      **step 7** (Excel download + preview). Steps 1–4: verbose log only (by design, for now).
- [x] Bilingual nl/en (labels via `ui_text.STEP_NAMES`, rest via a local `T()` helper).
- [x] **Pipeline prerequisite**: parameterized `run_taxonomy` / `run_codebook` /
      `run_assignment` to accept `filename/var_name/sample_size` (default = TEST_DATA, so
      `run_pipeline.py` is unchanged).
- [x] **Bug fix (default-arg capture)**: the load/save helpers in steps 4/5/6 had
      `filename = FILENAME` defaults bound once at import, so they ignored the entrypoint's
      `global` rebind → a run on a 2nd dataset silently loaded+saved the *first* dataset under
      its keys (new dataset stayed "locked"). Fixed: ~11 helpers now take `Optional[...] = None`
      and late-bind to the rebound globals in their body. Verified via unit test + AppTest.
- [x] **Bug fix**: `utils/saveVerbose.find_latest_log` glob never matched the on-disk filename
      (`{varkey}_{sample}_step{N}`) → verbose-log display was broken. Now lenient.
- [x] Fixed the local `~/.zshrc` auto-venv hook (subdirs like `src/` lost the venv on `cd`).

### 2026-06-09
- [x] **"Run all steps (1-7)" button** (sidebar, 2-step confirm). Backend `run_all_steps` is a
      generator that cascade-invalidates then force-reruns steps 1→7 sequentially, streaming a
      per-step ✅/❌ line via `st.status`, stopping on the first failure. No verbose interleaving:
      strictly sequential blocking runs + `sys.stdout.flush()` between steps. Lands on step 7 on
      success, the failed step otherwise.
- [x] **Dual export at the Export step (7)**. Step 7 now produces BOTH deliverables:
      `run_export` (results workbook `_codering.xlsx` + 4 `.sav`) AND `export_codebook`
      (codebook/taxonomy CSV + XLSX). `view_codebook.py`'s `__main__` was refactored into
      `export_codebook(filename, var_name, sample_size, *, write_csv, write_xlsx, print_console)`
      — standalone `python -m …view_codebook` is unchanged (prints readouts + writes files).
- [x] **Fix**: `run_export.run_step` now returns a **dict** of paths, and the results workbook was
      renamed `_code_assignments.xlsx` → `_codering.xlsx`. Updated `export_path()` + the step-7
      dispatch so the "done" probe (`is_step_done(7)`) and the results view find the file again.
- [x] **Fix (export path moved again)**: `resultsExporter` now writes to `exports/coderingen/`;
      `export_path` updated to match (was the old `exports/` root → step 7 falsely "not done").
- [x] **Fix (verbose-log retrieval)**: `saveVerbose.find_latest_log` picked the newest by
      filename, but on-disk names have two formats (`…_2500_2500_stepN_` vs `…_2500_stepN_`) so
      alphabetical ≠ chronological → wrong/older log shown per step. Now sorts by `mtime`.
- [x] **Fix (run-all re-entrancy)**: `page_run_all` kept `run_all=True` during the minutes-long
      blocking loop; an SSH/browser reconnect spawned a 2nd run that re-entered and called
      `invalidate_from(1)` again, wiping caches mid-run ("No taxonomy_codes cache" at step 7).
      Flag is now cleared up front → the loop runs exactly once.

---

## Audit: app_old → new app gaps (2026-06-09)

Systematic gap analysis (app_old was ~3410 lines, battle-tested; the new app is a fresh
rewrite, so most recent bugs are new-app bugs, not things app_old got wrong). Grouped by
priority. **Agreed work order:** P0 hardening → var_lab → codebook download → QA view →
quality-filter breakdown.

### P0 — drift couplings (the "keeps biting" class) — DO FIRST
The new app re-derives paths/names/keys that are actually *produced elsewhere*; if the source
changes, it breaks silently. The recent surprises (export path → `coderingen/`, verbose-log
returning the wrong file) are exactly this class.
- [ ] **Exporters return their canonical path; the app reads it back** instead of re-deriving.
      Applies to the results workbook (`export_path`) and the codebook XLSX. Source of truth =
      `resultsExporter` / `view_codebook`, not a mirrored string in `app_backend`.
- [ ] **One source of truth for step → cache-step-names → done-check.** `_STEP_DB_STEPS`
      and `is_step_done` (app_backend.py) independently hardcode the same magic strings
      (`taxonomy_codes`, `mece_codes`, `*_metadata`, …). Collapse into one per-step descriptor;
      ideally have runners expose their own cache step-name so the app imports it.
- [ ] (lower) the verbose-log glob and `variable_key` are also hand-mirrors — currently
      mitigated (mtime sort; shared key convention). Revisit only if they bite.

### P1 — functional regressions
- [ ] **var_lab editable + threaded through the pipeline.** The survey question is load-bearing
      LLM context: fix bad/garbled SPSS labels AND inject domain context (e.g. "eekhoorn = Merk X
      logo", bank names) so the LLM doesn't err. Now it's display-only and every runner re-fetches
      from SPSS. Add an edit field (upload + resume), `var_lab` on `StepConfig` (steps 1/3/7),
      `get_var_lab` prefers it (fallback SPSS), and editing it invalidates from step 1.
- [ ] **Cache-corruption recovery** (app_old `_load_or_recover`): on a corrupt / closed-file read,
      invalidate that row + recover instead of looping on the same error. (Do when it bites.)
- [ ] **Error persistence.** `last_run` error clears on the first rerun; keep it visible until
      acknowledged, translate it, add a remediation hint + link to the verbose log.

### P2 — user-value gaps
- [ ] **Codebook download** at step 7 (file already generated in `exports/codebook/`; just a
      button + a path helper). Note the full-sample `…_None.xlsx` naming quirk first.
- [ ] **QA drill-down at step 6**: respondent → ideas → assigned code + rationale + confidence,
      with re-roll. app_old's core "is this output trustworthy?" view; the aggregate tables in
      `render_results` don't replace it.
- [ ] **Quality-filter breakdown** (step 2): counts per category (meaningful / don't-know /
      no-response / gibberish) + %, from the cached `QualityFilteredModel.quality_filter_code`.
- [ ] **Result renderers for steps 1, 3, 4** (1: before/after sample; 3: response → ideas +
      abstraction ladder; 4: taxonomy tree). [deferred]
- [ ] **Per-step stat counts** in summaries (unique / single-vs-multi ideas; #domains/#facets/
      #attributes; responses/ideas/assigned) — currently static or minimal text.
- [ ] **Upload preview with metrics** (total / non-empty / unique / sample + first 10 rows)
      before spending LLM credits.
- [ ] **Text-variable validation** that *blocks* (not just filters) numeric selection on upload.

### Can stay dropped (old-pipeline-specific / cosmetic)
Multi-variable merge, advanced per-step model overrides, `st.balloons()`, the category/theme
sample browsers, file-size-MB in the cache list.

### Other known constraints
- [ ] **Long-running *single* steps block the UI** (run-all now streams via `st.status`; single
      steps still block behind a spinner). Background execution + live log tail is the real fix.
- [ ] **Full-sample codebook filename quirk**: `export_codebook` writes `…_None.xlsx` for a
      full-sample dataset; normalize to `full` (like `run_export`/`VerboseCapture`) before the download.
- [ ] **id_column on resume** defaults to `TEST_DATA`; only matters if step 0 is force-recomputed.
- [ ] **Cost display** — `token_tracker` summary (tokens + €) per run/step.
- [ ] **AppTest coverage** for the run / re-run / invalidate flows (not just render).

---

## Known constraints / gotchas

- The `global` rebind inside `run_taxonomy/run_codebook/run_assignment` is **process-global**.
  Fine for a local single-user app (runs are sequential), **not** safe for concurrent
  multi-user serving. If this ever goes multi-user, thread the dataset through explicitly
  instead of via module globals.
- Perf stats (`data/model_perf_stats.json`) are **dataset-scoped** (`filename:variable_key`)
  for steps 3, 5-p8, 6. A new dataset cold-starts those phases once (slower first run), then
  warms up — by design, not an app bug.
- `app.py` lazily imports the runner modules; **changing pipeline code requires a full
  Streamlit restart** (hot-reload won't reimport cached modules).
