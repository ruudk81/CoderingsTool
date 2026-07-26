"""
CoderingsTool — Streamlit UI (screen-model orchestrator over the cache-backed pipeline).

Architecture (app/dev/app_development_plan.md):
    The cache is the only truth for "done" (app_backend probes it live).
    The page layer: every step page makes ONE explicit screen decision —
    LOCKED | RUN | OUTPUT (| REVIEW, Phase D) — and pulls its step-specific
    content from the STEP_VIEWS registry in app_views.py. RUN screens explain
    the step before credits are spent; OUTPUT screens show evidence and offer
    the next step; errors stay visible until dismissed or superseded.

Run:  streamlit run src/app/app.py   (or ./run_app.sh)
"""

import os
import sys
import warnings
from pathlib import Path

import nest_asyncio
import streamlit as st

# src/ + src/app/ on path + cooperative event loop for the async runners
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
nest_asyncio.apply()
warnings.filterwarnings("ignore", message="To exit: use 'exit', 'quit', or Ctrl-D.")

import ui_text as ui
import app_backend as be
import app_views as av
from app_backend import DatasetSpec, LAST_STEP, Screen
from config import CacheConfig
from utils.cacheManager import CacheManager

st.set_page_config(page_title="CoderingsTool", page_icon="📊", layout="wide")

# =============================================================================
# Shared resources + session state
# =============================================================================

@st.cache_resource
def get_cache_manager() -> CacheManager:
    return CacheManager(CacheConfig())


def _concat_module():
    """concat_open_ends (concatenate/ is not a package — path-load it)."""
    concat_dir = str(be.PROJECT_ROOT / "concatenate")
    if concat_dir not in sys.path:
        sys.path.insert(0, concat_dir)
    import concat_open_ends
    return concat_open_ends

st.session_state.setdefault("step", 0)
st.session_state.setdefault("language", ui.DEFAULT_LANGUAGE)
st.session_state.setdefault("spec", None)           # DatasetSpec | None
st.session_state.setdefault("last_success", None)   # (step, summary) just executed
st.session_state.setdefault("last_error", None)     # (step, message) — STICKY
st.session_state.setdefault("epoch", 0)             # bumped on run/invalidate → busts view caches
st.session_state.setdefault("run_all", False)       # full-run view is active
st.session_state.setdefault("run_all_confirm", False)  # 2-step confirm armed

lang = st.session_state.language

def T(nl: str, en: str) -> str:
    """Tiny bilingual helper."""
    return nl if lang == "nl" else en

def step_name(step: int) -> str:
    return ui.get_text("STEP_NAMES", lang).get(step, be.STEP_LABELS[step])

def _bump_epoch():
    st.session_state.epoch += 1

# =============================================================================
# Run a step (blocking) with spinner; outcome lands in sticky session banners
# =============================================================================

def run_step(step: int, force_recalc: bool):
    spec = st.session_state.spec
    with st.spinner(T(f"Stap {step} draait…", f"Running step {step}…")):
        try:
            summary = be.run_step(step, spec, force_recalc=force_recalc)
            st.session_state.last_success = (step, summary)
            st.session_state.last_error = None   # a new successful run supersedes the error
        except Exception as exc:  # surface, don't crash the app
            st.session_state.last_error = (step, str(exc))
            st.session_state.last_success = None
    _bump_epoch()
    st.rerun()

# =============================================================================
# SIDEBAR — language, dataset, step navigator, cache management
# =============================================================================

def render_sidebar(status: dict, max_done: int):
    with st.sidebar:
        # Language
        names = {"Nederlands": "nl", "English": "en"}
        pick = st.selectbox("Taal / Language", list(names.keys()),
                            index=list(names.values()).index(lang))
        if names[pick] != lang:
            st.session_state.language = names[pick]
            st.rerun()

        spec = st.session_state.spec
        if spec is None:
            return

        st.divider()
        st.caption(f"**{Path(spec.filename).stem}**\n\n{spec.var_name} · "
                   f"{spec.sample_size if spec.sample_size is not None else T('volledig', 'full')}")
        if st.button(T("Andere dataset", "Change dataset"), width="stretch"):
            st.session_state.spec = None
            st.session_state.step = 0
            st.session_state.last_success = None
            st.session_state.last_error = None
            st.rerun()

        # Step navigator
        st.divider()
        st.markdown("**" + T("Stappen", "Steps") + "**")
        for s in range(0, LAST_STEP + 1):
            done = status[s]
            reachable = (s == 0) or done or (s <= max_done + 1)
            icon = "✅" if done else ("▶️" if reachable else "🔒")
            label = f"{icon} {s}. {step_name(s)}"
            if st.button(label, key=f"nav_{s}", width="stretch",
                         disabled=not reachable,
                         type="primary" if s == st.session_state.step else "secondary"):
                st.session_state.step = s
                st.rerun()

        # Cache management: cascade re-run
        st.divider()
        with st.expander(T("Cache / herverwerken", "Cache / reprocess")):
            cur = st.session_state.step
            st.caption(T(f"Herverwerken vanaf stap {cur} maakt stap {cur} t/m {LAST_STEP} ongeldig.",
                         f"Reprocessing from step {cur} invalidates steps {cur}–{LAST_STEP}."))
            if st.button(T(f"Herverwerk vanaf stap {cur}", f"Reprocess from step {cur}"),
                         width="stretch"):
                be.invalidate_from(cur, spec, get_cache_manager())
                st.session_state.last_success = None
                st.session_state.last_error = None
                _bump_epoch()
                st.toast(T(f"Cache gewist vanaf stap {cur}", f"Cache cleared from step {cur}"))
                st.rerun()

        # Run all steps 1-7 (2-step confirm: a full run costs minutes + LLM credits)
        with st.expander(T("Alles draaien (1-7)", "Run all (1-7)")):
            st.caption(T("Herberekent stap 1 t/m 7 volledig opnieuw. Dit kost meerdere "
                         "minuten en LLM-credits (€).",
                         "Fully recomputes steps 1-7. This takes several minutes and "
                         "LLM credits (€)."))
            if not st.session_state.run_all_confirm:
                if st.button(T("Alles opnieuw draaien", "Re-run all steps"),
                             width="stretch", key="run_all_arm"):
                    st.session_state.run_all_confirm = True
                    st.rerun()
            else:
                st.warning(T("Weet je het zeker? Stap 1-7 worden opnieuw berekend.",
                             "Are you sure? Steps 1-7 will be recomputed."))
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(T("Bevestig", "Confirm"), type="primary",
                                 width="stretch", key="run_all_go"):
                        st.session_state.run_all = True
                        st.session_state.run_all_confirm = False
                        st.session_state.last_success = None
                        st.session_state.last_error = None
                        st.rerun()
                with c2:
                    if st.button(T("Annuleer", "Cancel"),
                                 width="stretch", key="run_all_cancel"):
                        st.session_state.run_all_confirm = False
                        st.rerun()

# =============================================================================
# STEP 0 — upload / select dataset
# =============================================================================

def page_select_dataset():
    st.header("CoderingsTool")
    st.caption(ui.get_text("STEP_INFO", lang).get(0, ""))

    # --- Resume from cache ---
    st.subheader(T("Hervat uit cache", "Resume from cache"))
    specs = be.list_cached_datasets()
    if specs:
        labels = [s.display_name for s in specs]
        choice = st.selectbox(T("Beschikbare datasets", "Available datasets"),
                              options=range(len(specs)), format_func=lambda i: labels[i])
        chosen = specs[choice]
        cm = get_cache_manager()
        md = be.max_completed_step(chosen, cm)
        st.caption(T(f"Voltooid t/m stap {md} ({step_name(md)})",
                     f"Completed through step {md} ({step_name(md)})"))
        # The question is fixed once processing has started — the executed
        # steps already used it as LLM context. Read-only here; it is set at
        # the commit moment of a new dataset (§3.7).
        if chosen.var_lab:
            st.caption(T("Vraag", "Question") + f": _{chosen.var_lab}_")
        if st.button(T("Laden", "Load"), type="primary"):
            st.session_state.spec = chosen
            st.session_state.step = max(1, md)
            st.session_state.last_success = None
            st.session_state.last_error = None
            _bump_epoch()
            st.rerun()
    else:
        st.info(T("Geen datasets in cache.", "No cached datasets."))

    # --- Nieuw bestand: two-phase selection (plan §3.7) ---
    # Phase 1 (this page): choosing is LIGHT. One bounded read (inspect_sav,
    # ~9ms even on 71MB) feeds everything; nothing is written, so data/ stays
    # stable and no widget can be reset by the page's own side effects. A merge
    # is DECLARED here, not executed. Phase 2 (commit_selection): merge
    # write+verify, step 0 and the cache — all behind one status box.
    st.divider()
    st.subheader(T("Nieuw bestand", "New file"))
    server_files = sorted(p.name for p in (be.PROJECT_ROOT / "data").glob("*.sav"))
    src_pick = st.selectbox(T("Bestand op de server (data/)", "File on the server (data/)"),
                            server_files, index=None, key="server_pick",
                            placeholder=T("Kies een bestand…", "Pick a file…"))
    up = st.file_uploader(T("… of upload een nieuw SPSS-bestand (.sav)",
                            "… or upload a new SPSS file (.sav)"), type=["sav"])

    if src_pick:
        fname = src_pick
    elif up is not None:
        # Saving the upload is source acquisition, not derived work; once.
        if st.session_state.get("upload_saved") != up.name:
            dest = be.PROJECT_ROOT / "data" / up.name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(up.getbuffer())
            st.session_state.upload_saved = up.name
        fname = up.name
    else:
        return

    try:
        insp = be.inspect_sav(fname)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    all_vars = list(insp.variables)
    string_vars = insp.string_vars or all_vars

    intent = render_merge_intent(fname, insp, string_vars)

    col1, col2 = st.columns(2)
    with col1:
        id_col = st.selectbox(T("ID-kolom", "ID column"), all_vars, key=f"id_col_{fname}")
    with col2:
        if intent:
            # Declaring a merge means analyzing the merged variable.
            st.text_input(T("Tekstvariabele", "Text variable"),
                          value=intent["newvar"] + T(" (samengevoegd)", " (merged)"),
                          disabled=True, key=f"text_var_merged_{fname}")
            text_var = intent["newvar"]
        else:
            text_var = st.selectbox(T("Tekstvariabele", "Text variable"), string_vars,
                                    key=f"text_var_{fname}")

    # Keyed per file: un-keyed widgets can silently lose their state when a
    # rerun skips them — exactly how a chosen sample once vanished at commit.
    limit = st.checkbox(T("Steekproef beperken", "Limit sample"), value=False,
                        key=f"limit_{fname}")
    sample_size = st.number_input(T("Aantal", "Count"), min_value=10, max_value=100000,
                                  value=500, step=50,
                                  key=f"sample_{fname}") if limit else None

    # Survey question — editable LLM context. Prefill: the merge-inherited
    # question, else the picked variable's cleaned label. Metadata only.
    default_q = intent["question"] if intent else \
        be.clean_question(insp.variables[text_var]["label"], text_var)
    var_lab = st.text_area(
        T("Enquêtevraag (LLM-context)", "Survey question (LLM context)"),
        value=default_q or text_var, key=f"upload_varlab_{fname}_{text_var}", height=80,
        help=T("Corrigeer opmaak/spelling of voeg context toe (bv. 'de eekhoorn is het logo van merk X').",
               "Fix formatting/spelling or add context (e.g. 'the squirrel is brand X's logo)."))

    # Preview on demand, from the bounded frame (§3.7 decision 1)
    if st.toggle(T("Voorbeeld (eerste 200 rijen)", "Preview (first 200 rows)"),
                 key=f"preview_{fname}"):
        render_selection_preview(insp, intent, text_var)

    if st.button("🚀 " + T("Data laden (stap 0)", "Load data (step 0)"), type="primary"):
        commit_selection(fname, intent, text_var, sample_size, id_col, var_lab)


def render_merge_intent(fname: str, insp, string_vars: list):
    """§3.7: DECLARE a slot-series merge — nothing is written here.

    Returns {"newvar", "cols", "sep", "question"} for a valid intent, else None.
    Merge-integrity rule: every member must carry the same question (cleaned
    label); a mismatching series stays visible but is blocked, labels shown.
    The merged variable inherits the FIRST member's cleaned question."""
    groups = _concat_module().find_slot_groups(string_vars)
    if not groups:
        return None
    with st.expander(T("Variabelen samenvoegen (bijv. xQd1_1 … xQd1_10)",
                       "Merge variables (e.g. xQd1_1 … xQd1_10)")):
        st.caption(T("Meerkeuze-open-vragen staan vaak verspreid over genummerde "
                     "slots. Het samenvoegen zelf gebeurt pas bij 'Data laden'; "
                     "het bronbestand blijft onaangetast.",
                     "Multi-response open questions often sit in numbered slots. "
                     "The merge itself runs at 'Load data'; the source file "
                     "stays untouched."))
        fmt = {p: f"{cols[0]} … {cols[-1]} ({len(cols)})" for p, cols in groups.items()}
        none_label = T("— niet samenvoegen —", "— do not merge —")
        pick = st.selectbox(T("Reeks", "Series"), [None] + list(groups),
                            format_func=lambda p: none_label if p is None else fmt[p],
                            key=f"merge_pick_{fname}")
        if pick is None:
            return None
        cols = groups[pick]

        question, mismatches = be.series_question(insp, cols)
        if mismatches:
            st.error(T("Deze reeks draagt niet overal dezelfde vraag — "
                       "samenvoegen is geblokkeerd.",
                       "This series does not carry the same question throughout — "
                       "merging is blocked."))
            st.markdown(f"- `{cols[0]}` — _{question or '(leeg)'}_")
            for c, q in mismatches.items():
                st.markdown(f"- `{c}` — _{q or '(leeg)'}_")
            return None

        c1, c2 = st.columns(2)
        newvar = c1.text_input(T("Naam nieuwe variabele", "New variable name"),
                               value=pick.rstrip("_").lstrip("x") or pick,
                               key=f"merge_new_{fname}_{pick}").strip()
        sep = c2.text_input(T("Scheidingsteken", "Separator"), value=", ",
                            key=f"merge_sep_{fname}_{pick}")
        if not newvar:
            return None
        if newvar in insp.variables:
            st.error(T(f"'{newvar}' bestaat al in het bestand.",
                       f"'{newvar}' already exists in the file."))
            return None
        if question:
            st.caption(T(f"Vraag (geërfd van {cols[0]}): ",
                         f"Question (inherited from {cols[0]}): ") + f"_{question}_")
        return {"newvar": newvar, "cols": cols, "sep": sep, "question": question}


def render_selection_preview(insp, intent, text_var: str):
    """Sample of the (to-be-)analyzed variable from the bounded 200-row frame —
    a merge intent is previewed by combining the slots in memory."""
    if intent:
        co = _concat_module()
        values = insp.frame[intent["cols"]].apply(
            lambda r: co.combine_row(r, intent["sep"]), axis=1)
    else:
        values = insp.frame[text_var]
    s = values.fillna("").astype(str).str.strip()
    filled = s[s != ""]
    st.caption(f"{T('Eerste', 'First')} {len(s)} {T('rijen', 'rows')}: "
               f"{len(filled)} {T('gevuld', 'filled')} · "
               f"{filled.nunique()} {T('uniek', 'unique')}")
    for v in filled.head(8):
        st.markdown(f"- {v}")


def commit_selection(fname: str, intent, text_var: str, sample_size, id_col: str,
                     var_lab: str):
    """§3.7 phase 2: ALL heavy work at one moment, behind one status box —
    merge (reuse-when-fresh) → step 0 (load + cache) → navigate to step 1."""
    co = _concat_module()
    with st.status(T("Dataset vastleggen…", "Committing dataset…"),
                   expanded=True) as box:
        # Echo the exact identity being committed — a silently lost sample
        # choice must be visible HERE, before credits are spent downstream.
        size_txt = str(sample_size) if sample_size is not None else T("volledig", "full")
        st.write(T(f"Dataset: {text_var} · steekproef: {size_txt}",
                   f"Dataset: {text_var} · sample: {size_txt}"))
        if intent:
            src = be.PROJECT_ROOT / "data" / fname
            out = Path(co.default_outfile(str(src), intent["newvar"]))
            # Reuse-when-fresh (§3.7 decision 2): re-creating would touch the
            # mtime and needlessly invalidate that file's pipeline cache.
            reuse = out.exists() and out.stat().st_mtime > src.stat().st_mtime
            if reuse:
                try:
                    reuse = intent["newvar"] in be.inspect_sav(out.name).variables
                except RuntimeError:
                    reuse = False
            if reuse:
                st.write(T(f"Samengevoegd bestand hergebruikt: {out.name}",
                           f"Reusing merged file: {out.name}"))
            else:
                st.write(T("Variabelen samenvoegen…", "Merging variables…"))
                try:
                    res = co.concat_variables(str(src), intent["newvar"],
                                              vars_list=intent["cols"],
                                              sep=intent["sep"],
                                              label=intent["question"] or None)
                except (ValueError, RuntimeError) as exc:
                    box.update(label=T("Samenvoegen mislukt", "Merge failed"),
                               state="error")
                    st.error(str(exc))
                    return
                st.write(T(f"Geschreven en geverifieerd: {Path(res['outfile']).name} "
                           f"({res['filled']}/{res['rows']} rijen gevuld)",
                           f"Written and verified: {Path(res['outfile']).name} "
                           f"({res['filled']}/{res['rows']} rows filled)"))
            fname = out.name

        st.write(T("Data laden (stap 0)…", "Loading data (step 0)…"))
        spec = DatasetSpec(filename=fname, var_name=text_var,
                           sample_size=sample_size, id_column=id_col,
                           var_lab=(var_lab or "").strip() or text_var)
        try:
            be.run_step(0, spec, force_recalc=False)
        except Exception as exc:
            box.update(label=T("Laden mislukt", "Load failed"), state="error")
            st.error(str(exc))
            return
        # Commit fixes the question — also on a step-0 cache hit, which would
        # otherwise leave an older (or absent) question on the data row.
        be.set_question(spec)
        box.update(label=T("Dataset geladen", "Dataset loaded"), state="complete")

    st.session_state.spec = spec
    st.session_state.step = 1
    _bump_epoch()
    st.rerun()

# =============================================================================
# STEPS 1-7 — one explicit screen decision, content from the registry
# =============================================================================

def render_banners(step: int):
    """Sticky outcome banners. Errors persist until dismissed or superseded."""
    err = st.session_state.last_error
    if err and err[0] == step:
        c1, c2 = st.columns([6, 1])
        with c1:
            st.error(err[1])
            st.caption(T("Zie het uitvoeringslog hieronder voor details.",
                         "See the execution log below for details."))
        with c2:
            if st.button("✖️", key=f"dismiss_err_{step}",
                         help=T("Melding sluiten", "Dismiss")):
                st.session_state.last_error = None
                st.rerun()

    ok = st.session_state.last_success
    if ok and ok[0] == step:
        summary = ok[1]
        if "⚠️ WAARSCHUWING" in summary:
            st.warning(summary)
        else:
            st.success(summary)


def render_locked(step: int):
    st.warning(T(f"Voltooi eerst stap {step - 1} ({step_name(step - 1)}).",
                         f"Complete step {step - 1} ({step_name(step - 1)}) first."))


def render_run(step: int):
    """RUN screen: explain what will happen, then offer the button."""
    info = ui.get_text("STEP_INFO", lang).get(step)
    if info:
        st.info(info)
    model = av.models_line(step)
    if model:
        st.caption(T("Model", "Model") + f": {model}")
    if st.button("🚀 " + T(f"Draai stap {step}", f"Run step {step}"),
                 type="primary", key=f"run_{step}"):
        run_step(step, force_recalc=False)


def render_output(step: int, spec: DatasetSpec):
    """OUTPUT screen: evidence in tabs first, the way forward below it (calm:
    review the result, then advance — one thing in view at a time,
    Resultaat | Steekproef | Rapport)."""
    av.render_cost_line(spec, lang, step)

    view = av.STEP_VIEWS[step]
    epoch = st.session_state.epoch
    log = be.find_verbose_log(spec, step)
    labels = [T("Resultaat", "Result")]
    if view.samples:
        labels.append(T("Steekproef", "Sample"))
    if log:
        labels.append(T("Rapport", "Report"))
    tabs = st.tabs(labels)
    with tabs[0]:
        if view.stats:
            view.stats(spec, lang, epoch)
    if view.samples:
        with tabs[labels.index(T("Steekproef", "Sample"))]:
            view.samples(spec, lang, epoch)
    if log:
        with tabs[-1]:
            # The raw execution log, as-is. The Resultaat tab owns the distilled
            # view; this tab is the honest, complete record (monospace = correct
            # for terminal output). Stamp it so an older run's log is recognizable.
            ts = be.verbose_log_time(spec, step)
            if ts:
                st.caption(T(f"Log van {ts}", f"Log from {ts}"))
            st.code(log, language=None)

    # Actions below the evidence: review first, then advance.
    st.divider()
    c1, c2, _ = st.columns([1, 2, 3])
    with c1:
        if st.button(T("Opnieuw", "Re-run"), key=f"rerun_{step}"):
            be.invalidate_from(step, spec, get_cache_manager())
            run_step(step, force_recalc=True)
    with c2:
        if step < LAST_STEP:
            if st.button(T(f"Volgende: {step_name(step + 1)} ▶",
                           f"Next: {step_name(step + 1)} ▶"),
                         type="primary", key=f"continue_{step}"):
                st.session_state.step = step + 1
                st.session_state.last_success = None
                st.rerun()


def page_step(step: int, status: dict):
    spec = st.session_state.spec
    st.header(f"{step}. {step_name(step)}")
    st.caption(f"**Data:** {spec.var_name} · "
               f"{spec.sample_size if spec.sample_size is not None else T('volledig', 'full')}")
    # The survey question is fixed context during a run — editable only when
    # loading a dataset (step 0), read-only here so every screen shows what
    # the LLM was told the data is about.
    if spec.var_lab:
        st.caption(T("Vraag", "Question") + f": _{spec.var_lab}_")

    render_banners(step)

    screen = be.screen_for(step, status)
    if screen is Screen.LOCKED:
        render_locked(step)
    elif screen is Screen.RUN:
        render_run(step)
    else:  # OUTPUT (REVIEW arrives in Phase D)
        render_output(step, spec)

# =============================================================================
# RUN ALL — sequential force-recompute of steps 1-7 (full-width, streamed)
# =============================================================================

def page_run_all():
    spec = st.session_state.spec
    cm = get_cache_manager()
    # Clear the flag UP FRONT — this run owns the loop. The loop blocks for minutes;
    # if the SSH tunnel/browser reconnects meanwhile, Streamlit starts a second script
    # run that would see run_all=True, re-enter here, and call invalidate_from(1) AGAIN,
    # wiping caches out from under the in-flight run. With the flag already false, any
    # concurrent rerun renders the normal page and the loop runs exactly once.
    st.session_state.run_all = False
    st.header(T("Alle stappen draaien (1-7)", "Running all steps (1-7)"))
    st.caption(T("Samenvatting per stap hieronder.",
                 "Per-step summary below."))

    failed_step = None
    with st.status(T("Bezig met stap 1-7…", "Running steps 1-7…"),
                   expanded=True) as status_box:
        for res in be.run_all_steps(spec, cm):
            if res.ok:
                status_box.update(label=T(f"Stap {res.step} ({step_name(res.step)}) klaar",
                                          f"Step {res.step} ({step_name(res.step)}) done"))
                st.write(f"✅ **{res.step}. {step_name(res.step)}** — {res.summary}")
            else:
                st.write(f"❌ **{res.step}. {step_name(res.step)}** — "
                         f"{res.summary.replace('__ERROR__', '')}")
                failed_step = res.step
        if failed_step is None:
            status_box.update(label=T("Alle stappen voltooid ✅", "All steps complete ✅"),
                              state="complete")
        else:
            status_box.update(label=T(f"Gestopt bij stap {failed_step} ❌",
                                      f"Stopped at step {failed_step} ❌"), state="error")

    _bump_epoch()
    if failed_step is None:
        st.session_state.step = LAST_STEP
        st.toast(T("Pipeline voltooid — ga naar Export.", "Pipeline complete — see Export."))
    else:
        st.session_state.step = failed_step
        st.toast(T(f"Mislukt bij stap {failed_step}.", f"Failed at step {failed_step}."))
    # Manual continue (not auto-rerun) so the finished summary stays readable.
    if st.button(T("Doorgaan", "Continue"), type="primary", key="run_all_done"):
        st.rerun()

# =============================================================================
# MAIN
# =============================================================================

spec = st.session_state.spec
if spec is None:
    render_sidebar({}, -1)
    page_select_dataset()
else:
    cm = get_cache_manager()
    status = be.step_status(spec, cm)
    max_done = be.max_completed_step(spec, cm)
    render_sidebar(status, max_done)
    if st.session_state.run_all:
        page_run_all()
    elif st.session_state.step == 0:
        page_select_dataset()
    else:
        page_step(st.session_state.step, status)
