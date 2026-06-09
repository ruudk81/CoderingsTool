"""
CoderingsTool — Streamlit UI (orchestrator over the cache-backed pipeline).

Architecture (see app_backend.py for the why):
    The cache is the source of truth. This file is UI only: a step wizard that
    reads live "done" status from the cache, runs a step's pipeline runner on
    demand, shows the captured verbose log, and offers cascade re-runs. Results
    are read back from the cache and the step-7 Excel export.

Run:  cd src && streamlit run app.py
"""

import os
import sys
import warnings
from pathlib import Path

import nest_asyncio
import streamlit as st

# src/ on path + cooperative event loop for the async runners
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
nest_asyncio.apply()
warnings.filterwarnings("ignore", message="To exit: use 'exit', 'quit', or Ctrl-D.")

import ui_text as ui
import app_backend as be
from app_backend import DatasetSpec, LAST_STEP
from config import CacheConfig
from utils.cacheManager import CacheManager
from utils.dataLoader import DataLoader

st.set_page_config(page_title="CoderingsTool", page_icon="📊", layout="wide")

# =============================================================================
# Shared resources + session state
# =============================================================================

@st.cache_resource
def get_cache_manager() -> CacheManager:
    return CacheManager(CacheConfig())

@st.cache_resource
def get_data_loader() -> DataLoader:
    return DataLoader(data_dir=str(be.PROJECT_ROOT / "data"), verbose=False)

st.session_state.setdefault("step", 0)
st.session_state.setdefault("language", ui.DEFAULT_LANGUAGE)
st.session_state.setdefault("spec", None)          # DatasetSpec | None
st.session_state.setdefault("last_run", None)       # (step, summary) just executed
st.session_state.setdefault("run_all", False)       # full-run view is active
st.session_state.setdefault("run_all_confirm", False)  # 2-step confirm armed

lang = st.session_state.language

def T(nl: str, en: str) -> str:
    """Tiny bilingual helper."""
    return nl if lang == "nl" else en

def step_name(step: int) -> str:
    return ui.get_text("STEP_NAMES", lang).get(step, be.STEP_LABELS[step])

# =============================================================================
# Run a step (blocking) with spinner + transient feedback
# =============================================================================

def run_step(step: int, force_recalc: bool):
    spec = st.session_state.spec
    with st.spinner(T(f"Stap {step} draait… (live voortgang in de terminal)",
                      f"Running step {step}… (live progress in the terminal)")):
        try:
            summary = be.run_step(step, spec, force_recalc=force_recalc)
            st.session_state.last_run = (step, summary)
        except Exception as exc:  # surface, don't crash the app
            st.session_state.last_run = (step, f"__ERROR__ {exc}")
    st.rerun()

# =============================================================================
# SIDEBAR — language, dataset, step navigator, cache management
# =============================================================================

def render_sidebar(status: dict, max_done: int):
    with st.sidebar:
        # Language
        names = {"Nederlands": "nl", "English": "en"}
        pick = st.selectbox("🌐 Taal / Language", list(names.keys()),
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
        if st.button("🏠 " + T("Andere dataset", "Change dataset"), width="stretch"):
            st.session_state.spec = None
            st.session_state.step = 0
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
        with st.expander("🔧 " + T("Cache / herverwerken", "Cache / reprocess")):
            cur = st.session_state.step
            st.caption(T(f"Herverwerken vanaf stap {cur} maakt stap {cur} t/m {LAST_STEP} ongeldig.",
                         f"Reprocessing from step {cur} invalidates steps {cur}–{LAST_STEP}."))
            if st.button("🔄 " + T(f"Herverwerk vanaf stap {cur}", f"Reprocess from step {cur}"),
                         width="stretch"):
                be.invalidate_from(cur, spec, get_cache_manager())
                st.session_state.last_run = None
                st.toast(T(f"Cache gewist vanaf stap {cur}", f"Cache cleared from step {cur}"))
                st.rerun()

        # Run all steps 1-7 (2-step confirm: a full run costs minutes + LLM credits)
        with st.expander("⏩ " + T("Alles draaien (1-7)", "Run all (1-7)")):
            st.caption(T("Herberekent stap 1 t/m 7 volledig opnieuw. Dit kost meerdere "
                         "minuten en LLM-credits (€).",
                         "Fully recomputes steps 1-7. This takes several minutes and "
                         "LLM credits (€)."))
            if not st.session_state.run_all_confirm:
                if st.button("⏩ " + T("Alles opnieuw draaien", "Re-run all steps"),
                             width="stretch", key="run_all_arm"):
                    st.session_state.run_all_confirm = True
                    st.rerun()
            else:
                st.warning(T("Weet je het zeker? Stap 1-7 worden opnieuw berekend.",
                             "Are you sure? Steps 1-7 will be recomputed."))
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ " + T("Bevestig", "Confirm"), type="primary",
                                 width="stretch", key="run_all_go"):
                        st.session_state.run_all = True
                        st.session_state.run_all_confirm = False
                        st.session_state.last_run = None
                        st.rerun()
                with c2:
                    if st.button("✖️ " + T("Annuleer", "Cancel"),
                                 width="stretch", key="run_all_cancel"):
                        st.session_state.run_all_confirm = False
                        st.rerun()

# =============================================================================
# STEP 0 — upload / select dataset
# =============================================================================

def page_select_dataset():
    st.header("📊 CoderingsTool")
    st.caption(T("Selecteer een eerder verwerkte dataset of upload een nieuw SPSS-bestand.",
                 "Pick a previously processed dataset or upload a new SPSS file."))

    # --- Resume from cache ---
    st.subheader("📂 " + T("Hervat uit cache", "Resume from cache"))
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
        if st.button("📂 " + T("Laden", "Load"), type="primary"):
            st.session_state.spec = chosen
            st.session_state.step = max(1, md)
            st.session_state.last_run = None
            st.rerun()
    else:
        st.info(T("Geen datasets in cache.", "No cached datasets."))

    # --- Upload new ---
    st.divider()
    st.subheader("📤 " + T("Nieuw bestand", "New file"))
    up = st.file_uploader(T("Kies een SPSS-bestand (.sav)", "Choose an SPSS file (.sav)"),
                          type=["sav"])
    if up is not None:
        dest = be.PROJECT_ROOT / "data" / up.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(up.getbuffer())
        loader = get_data_loader()
        try:
            var_types = loader.list_variables_with_types(up.name)
        except Exception as exc:
            st.error(T(f"Kon variabelen niet lezen: {exc}", f"Could not read variables: {exc}"))
            return

        all_vars = list(var_types.keys())
        string_vars = [v for v, i in var_types.items() if i.get("is_string")] or all_vars

        col1, col2 = st.columns(2)
        with col1:
            id_col = st.selectbox("🆔 " + T("ID-kolom", "ID column"), all_vars)
        with col2:
            text_var = st.selectbox("📄 " + T("Tekstvariabele", "Text variable"), string_vars)

        limit = st.checkbox(T("Steekproef beperken", "Limit sample"), value=False)
        sample_size = st.number_input(T("Aantal", "Count"), min_value=10, max_value=100000,
                                      value=500, step=50) if limit else None

        # Survey question — editable LLM context (fix typos/formatting, inject domain context).
        try:
            spss_lab = loader.get_varlab(up.name, text_var)
            spss_lab = spss_lab[spss_lab.rfind("]") + 1:].strip()
        except Exception:
            spss_lab = text_var
        var_lab = st.text_area(
            "📝 " + T("Enquêtevraag (LLM-context)", "Survey question (LLM context)"),
            value=spss_lab, key=f"upload_varlab_{text_var}", height=80,
            help=T("Corrigeer opmaak/spelling of voeg context toe (bv. 'de eekhoorn is het logo van Merk X').",
                   "Fix formatting/spelling or add context (e.g. 'the squirrel is Merk X's logo')."))

        if st.button("🚀 " + T("Data laden (stap 0)", "Load data (step 0)"), type="primary"):
            spec = DatasetSpec(filename=up.name, var_name=text_var,
                               sample_size=sample_size, id_column=id_col,
                               var_lab=(var_lab or "").strip() or text_var)
            st.session_state.spec = spec
            be.run_step(0, spec, force_recalc=False)  # load + cache
            st.session_state.step = 1
            st.rerun()

# =============================================================================
# STEPS 1-7 — generic step page
# =============================================================================

def page_step(step: int, status: dict, max_done: int):
    spec = st.session_state.spec
    st.header(f"{step}. {step_name(step)}")

    st.caption(f"**Data:** {spec.var_name} · "
               f"{spec.sample_size if spec.sample_size is not None else T('volledig', 'full')}")
    # The survey question is LLM context (spell-check + extraction + classification).
    # It's editable; applying a change re-runs from step 1 (where the context first matters).
    _vk = f"varlab_{spec.variable_key}"
    st.session_state.setdefault(_vk, spec.var_lab or "")
    with st.expander("📝 " + T("Enquêtevraag / context", "Survey question / context")):
        edited = st.text_area(
            T("Vraag (LLM-context — corrigeer opmaak/spelling of voeg context toe)",
              "Question (LLM context — fix formatting/spelling or add context)"),
            key=_vk, height=80,
            help=T("Bv. 'de eekhoorn is het logo van Merk X'. Toepassen herverwerkt vanaf stap 1.",
                   "E.g. 'the squirrel is Merk X's logo'. Applying reprocesses from step 1."))
        if edited.strip() != (spec.var_lab or "").strip():
            if st.button("💾 " + T("Toepassen (herverwerk vanaf stap 1)",
                                   "Apply (reprocess from step 1)"), key="apply_varlab"):
                spec.var_lab = edited.strip()
                be.invalidate_from(1, spec, get_cache_manager())
                st.session_state.last_run = None
                st.toast(T("Vraag bijgewerkt — draai opnieuw vanaf stap 1.",
                           "Question updated — re-run from step 1."))
                st.rerun()

    prev_done = (step == 0) or status.get(step - 1, False)
    done = status[step]

    # Transient feedback from a run that just finished
    if st.session_state.last_run and st.session_state.last_run[0] == step:
        summary = st.session_state.last_run[1]
        if summary.startswith("__ERROR__"):
            st.error(summary.replace("__ERROR__", "❌"))
        else:
            st.success(f"✅ {summary}")
        st.session_state.last_run = None

    # Run / re-run controls
    if not prev_done:
        st.warning(T(f"Voltooi eerst stap {step - 1} ({step_name(step - 1)}).",
                     f"Complete step {step - 1} ({step_name(step - 1)}) first."))
    elif not done:
        st.markdown(T("Klaar om te draaien.", "Ready to run."))
        if st.button("🚀 " + T(f"Draai stap {step}", f"Run step {step}"), type="primary"):
            run_step(step, force_recalc=False)
    else:
        st.success("✅ " + T("Voltooid (uit cache).", "Completed (from cache)."))
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("🔄 " + T("Opnieuw", "Re-run"), key=f"rerun_{step}"):
                be.invalidate_from(step, spec, get_cache_manager())
                run_step(step, force_recalc=True)

    # Verbose execution log
    log = be.find_verbose_log(spec, step)
    if log:
        with st.expander("📋 " + T("Uitvoeringslog", "Execution log")):
            st.code(log, language=None)

    # Results
    if done:
        render_results(step, spec)


def render_results(step: int, spec: DatasetSpec):
    """Lazy, light result views. Rich tabular output lives in the step-7 export."""
    if step == 5:
        codes = be.load_codebook(spec)
        if codes and codes.raw_codes:
            st.subheader(T("Codeboek", "Codebook") + f" ({len(codes.raw_codes)})")
            rows = [{"code": c.get("code_name", ""), "valence": c.get("valence", ""),
                     "definition": c.get("definition", "")} for c in codes.raw_codes]
            st.dataframe(rows, width="stretch", hide_index=True)

    elif step == 6:
        models = be.load_assignments(spec)
        if models:
            import collections
            counter = collections.Counter()
            for m in models:
                for idea in (m.response_ideas or []):
                    if idea.assigned_code:
                        counter[idea.assigned_code] += 1
            st.subheader(T("Codefrequenties", "Code frequencies"))
            st.dataframe([{"code": c, "n": n} for c, n in counter.most_common()],
                         width="stretch", hide_index=True)

    elif step == 7:
        st.subheader(T("Export", "Export"))
        _xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        path = be.export_path(spec)        # results workbook
        cb = be.codebook_path(spec)        # codebook workbook
        c1, c2 = st.columns(2)
        with c1:
            if path.exists():
                st.download_button("⬇️ " + T("Resultaten (Excel)", "Results (Excel)"),
                                   data=path.read_bytes(), file_name=path.name,
                                   mime=_xlsx_mime, width="stretch")
        with c2:
            if cb.exists():
                st.download_button("⬇️ " + T("Codeboek (Excel)", "Codebook (Excel)"),
                                   data=cb.read_bytes(), file_name=cb.name,
                                   mime=_xlsx_mime, width="stretch")
        if path.exists():
            try:
                import pandas as pd
                # mixed-type columns → str so Arrow can render the preview
                df = pd.read_excel(path).astype(str)
                st.caption(f"{len(df)} {T('rijen', 'rows')} · {len(df.columns)} {T('kolommen', 'columns')}")
                st.dataframe(df.head(50), width="stretch", hide_index=True)
            except Exception as exc:
                st.caption(T(f"Voorbeeld niet beschikbaar: {exc}", f"Preview unavailable: {exc}"))

# =============================================================================
# RUN ALL — sequential force-recompute of steps 1-7 (full-width, streamed)
# =============================================================================

def page_run_all():
    spec = st.session_state.spec
    cm = get_cache_manager()
    # Clear the flag UP FRONT — this run owns the loop. The loop blocks for minutes;
    # if the SSH tunnel/browser reconnects meanwhile, Streamlit starts a second script
    # run that would see run_all=True, re-enter here, and call invalidate_from(1) AGAIN,
    # wiping caches out from under the in-flight run (e.g. "No taxonomy_codes cache" at
    # step 7). With the flag already false, any concurrent rerun renders the normal page
    # and the loop runs exactly once.
    st.session_state.run_all = False
    st.header("⏩ " + T("Alle stappen draaien (1-7)", "Running all steps (1-7)"))
    st.caption(T("Live voortgang in de terminal; samenvatting per stap hieronder.",
                 "Live progress in the terminal; per-step summary below."))

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
        page_step(st.session_state.step, status, max_done)
