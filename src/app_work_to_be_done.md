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

---

## To be done (next)

### High priority
- [ ] **Long-running steps block the UI.** A step (3/4/5/6) runs synchronously behind a
      spinner; live progress only shows in the terminal/verbose log. Options: run the step in
      a background thread/process and stream progress, or at least a `st.status()` with periodic
      log tailing. (User flagged this on day 1.)
- [ ] **Result renderers for steps 1–4** (deferred on purpose):
  - step 1 — sample of before/after preprocessed text.
  - step 2 — counts per filter category (meaningful / empty / don't-know / gibberish).
  - step 3 — sample responses with extracted ideas + abstraction ladder.
  - step 4 — taxonomy tree (domain → facet → attribute, with valence).

### Medium priority
- [ ] **Error persistence.** A failed run shows a transient `last_run` error that disappears on
      the next rerun. Persist/surface it (and link to the verbose log) until acknowledged.
- [ ] **Cost display.** Show `token_tracker` summary (tokens + €) per run, per step.
- [ ] **Advanced settings panel.** `app_old.py` had per-step model/param overrides in the
      sidebar; not ported. Decide what (if anything) to expose for the new pipeline.
- [ ] **id_column on resume.** For datasets loaded from cache, `id_column` defaults to
      `TEST_DATA.id_column`; only matters if step 0 is force-recalculated (re-reads SPSS).
      Consider persisting it (it's not in `cache_metadata`).

### Lower priority / later
- [ ] **Multi-variable merge.** New pipeline is single-variable only (test data uses a
      pre-combined `Qd1_combined`); `app_old.py` supported selecting+merging multiple SPSS vars.
      Re-add only if the pipeline grows it.
- [ ] **Dedicated results page** beyond step 7 (cross-step summary / export browser).
- [ ] **AppTest coverage** for the run/re-run/invalidate flows (not just render).

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
