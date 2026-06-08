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

        if st.button("🚀 " + T("Data laden (stap 0)", "Load data (step 0)"), type="primary"):
            try:
                var_lab = loader.get_varlab(up.name, text_var)
                var_lab = var_lab[var_lab.rfind("]") + 1:].strip()
            except Exception:
                var_lab = text_var
            spec = DatasetSpec(filename=up.name, var_name=text_var,
                               sample_size=sample_size, id_column=id_col, var_lab=var_lab)
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

    if spec.var_lab:
        st.info(f"**{T('Vraag', 'Question')}:** {spec.var_lab}  \n"
                f"**Data:** {spec.var_name} · "
                f"{spec.sample_size if spec.sample_size is not None else T('volledig', 'full')}")

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
        path = be.export_path(spec)
        if path.exists():
            st.subheader(T("Export", "Export"))
            st.download_button("⬇️ " + T("Download Excel", "Download Excel"),
                               data=path.read_bytes(), file_name=path.name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            try:
                import pandas as pd
                df = pd.read_excel(path)
                st.caption(f"{len(df)} {T('rijen', 'rows')} · {len(df.columns)} {T('kolommen', 'columns')}")
                st.dataframe(df.head(50), width="stretch", hide_index=True)
            except Exception as exc:
                st.caption(T(f"Voorbeeld niet beschikbaar: {exc}", f"Preview unavailable: {exc}"))

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
    if st.session_state.step == 0:
        page_select_dataset()
    else:
        page_step(st.session_state.step, status, max_done)
