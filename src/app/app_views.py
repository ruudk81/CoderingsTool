"""
app_views.py — per-step view registry for the Streamlit app (app_v2).

Each step contributes a StepView: which LLM phases it runs (for the read-only
model line on the RUN screen), and how to render its evidence on the OUTPUT
screen (stats panel, samples panel). Adding or enriching a step's presentation
is a registry entry here — page logic in app_v2.py never changes for it.
(Design: utils/dev/app_development_plan.md §3.4; HITL `review` slot is wired
for Phase D and unused until then.)

Loader results are cached with @st.cache_data, keyed on the dataset identity
plus an `epoch` that app_v2 bumps after every run/invalidation — so a re-run
busts the cache without TTL guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import streamlit as st

import app_backend as be
from app_backend import DatasetSpec
from config import get_step_model


def _t(lang: str, nl: str, en: str) -> str:
    return nl if lang == "nl" else en


# =============================================================================
# REGISTRY
# =============================================================================

# A renderer draws its panel directly with st.*: (spec, lang, epoch) -> None
Renderer = Callable[[DatasetSpec, str, int], None]


@dataclass(frozen=True)
class StepView:
    """Presentation contract of one pipeline step."""
    phases: tuple = ()                    # config.STEP_MODEL_TIERS keys this step runs
    stats: Optional[Renderer] = None      # OUTPUT: aggregate evidence panel
    samples: Optional[Renderer] = None    # OUTPUT: sample/inspection panel
    review: Optional[Renderer] = None     # HITL editable view (Phase D)


def models_line(step: int) -> Optional[str]:
    """Read-only 'which model(s) run this step' line, derived from config.py."""
    phases = STEP_VIEWS[step].phases
    if not phases:
        return None
    models = list(dict.fromkeys(get_step_model(p) for p in phases))  # unique, ordered
    return ", ".join(models)


# =============================================================================
# CACHED LOADERS — dataset identity + epoch as the cache key
# =============================================================================

def _spec(filename: str, var_name: str, sample_size: Optional[int]) -> DatasetSpec:
    return DatasetSpec(filename=filename, var_name=var_name, sample_size=sample_size)


@st.cache_data(max_entries=16, show_spinner=False)
def _quality_filtered(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_quality_filtered(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _codebook(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_codebook(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _assignments(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_assignments(_spec(filename, var_name, sample_size))


# =============================================================================
# VERBOSE REPORT (Phase B0) — render a parsed execution log as a report:
# sections as headers, summaries highlighted, telemetry behind a toggle.
# =============================================================================

def render_log_report(log_text: str, lang: str, key: str):
    rep = be.parse_verbose_log(log_text)

    if rep.meta:
        bits = [rep.meta[k] for k in ("Variable", "Sample size") if k in rep.meta]
        times = " → ".join(rep.meta[k] for k in ("Start time", "End time") if k in rep.meta)
        st.caption(" · ".join(bits + ([times] if times else [])))

    for sec in rep.sections:
        if sec.title:
            st.markdown(f"**{sec.title}**")
        if sec.body:
            st.text("\n".join(sec.body))
        if sec.summary:
            st.code("\n".join(sec.summary), language=None)

    if rep.noise_count and st.toggle(
            "⚙️ " + _t(lang, f"Technische details ({rep.noise_count} regels)",
                       f"Technical details ({rep.noise_count} lines)"),
            key=f"{key}_noise"):
        st.code("\n".join(ln for s in rep.sections for ln in s.noise), language=None)

    if st.toggle("📄 " + _t(lang, "Ruwe log", "Raw log"), key=f"{key}_raw"):
        st.code(log_text, language=None)


# =============================================================================
# STEP 2 — quality-filter breakdown
# =============================================================================

def stats_quality_filter(spec: DatasetSpec, lang: str, epoch: int):
    import collections
    data = _quality_filtered(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not data:
        return
    # Meaningful responses carry no filter code (None); 0 also = meaningful.
    labels = {None: _t(lang, "Betekenisvol", "Meaningful"),
              0: _t(lang, "Betekenisvol", "Meaningful"),
              99999997: _t(lang, "Weet niet / geen mening", "Don't know"),
              99999998: _t(lang, "Geen antwoord / leeg", "No answer / empty"),
              99999999: _t(lang, "Betekenisloos", "Gibberish")}
    counts = collections.Counter(getattr(d, "quality_filter_code", None) for d in data)
    total = len(data) or 1
    st.subheader(_t(lang, "Kwaliteitsfilter — uitsplitsing", "Quality filter — breakdown"))
    rows = [{_t(lang, "categorie", "category"): labels.get(code, str(code)),
             "n": n, "%": f"{100 * n / total:.1f}"}
            for code, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    st.dataframe(rows, width="stretch", hide_index=True)


# =============================================================================
# STEP 5 — codebook table
# =============================================================================

def stats_codebook(spec: DatasetSpec, lang: str, epoch: int):
    codes = _codebook(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not (codes and codes.raw_codes):
        return
    st.subheader(_t(lang, "Codeboek", "Codebook") + f" ({len(codes.raw_codes)})")
    rows = [{"code": c.get("code_name", ""), "valence": c.get("valence", ""),
             "definition": c.get("definition", "")} for c in codes.raw_codes]
    st.dataframe(rows, width="stretch", hide_index=True)


# =============================================================================
# STEP 6 — code frequencies + respondent QA drill-down
# =============================================================================

def stats_assignments(spec: DatasetSpec, lang: str, epoch: int):
    import collections
    models = _assignments(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not models:
        return
    # Count via assigned_code_id (K#) so the codebook's CURRENT name wins after
    # a rename; ideas without an id fall back to their stored name.
    codes = _codebook(spec.filename, spec.var_name, spec.sample_size, epoch)
    id_to_name = {c["code_id"]: c["code_name"]
                  for c in (codes.raw_codes if codes else [])
                  if c.get("code_id")}
    counter = collections.Counter()
    for m in models:
        for idea in (m.response_ideas or []):
            label = id_to_name.get(idea.assigned_code_id) or idea.assigned_code
            if label:
                counter[label] += 1
    st.subheader(_t(lang, "Codefrequenties", "Code frequencies"))
    st.dataframe([{"code": c, "n": n} for c, n in counter.most_common()],
                 width="stretch", hide_index=True)


def samples_assignments_qa(spec: DatasetSpec, lang: str, epoch: int):
    """QA drill-down: one respondent's assignments + rationale ("is this right?")."""
    import random
    models = _assignments(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not models:
        return
    coded = [m for m in models if any(i.assigned_code for i in (m.response_ideas or []))]
    if not coded:
        return
    st.divider()
    st.subheader("🔍 " + _t(lang, "Inspecteer een respondent", "Inspect a respondent"))
    rk = f"qa6_{spec.variable_key}"
    if rk not in st.session_state:
        st.session_state[rk] = random.randrange(len(coded))
    if st.button("🎲 " + _t(lang, "Andere respondent", "Another respondent"), key="qa6_roll"):
        st.session_state[rk] = random.randrange(len(coded))
    m = coded[st.session_state[rk] % len(coded)]
    st.markdown(f"**{_t(lang, 'Respondent', 'Respondent')}:** `{m.respondent_id}`")
    st.markdown(f"> {m.response}")
    for i in (m.response_ideas or []):
        if not i.assigned_code:
            continue
        conf = f" · {i.confidence:.2f}" if i.confidence is not None else ""
        st.markdown(f"- **{i.assigned_code}**{conf} — _{i.idea or i.instance}_")
        if i.rationale:
            st.caption(f"&nbsp;&nbsp;&nbsp;↳ {i.rationale}")


# =============================================================================
# STEP 7 — export downloads + preview
# =============================================================================

def stats_export(spec: DatasetSpec, lang: str, epoch: int):
    st.subheader(_t(lang, "Export", "Export"))
    _xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    path = be.export_path(spec)        # results workbook
    cb = be.codebook_path(spec)        # codebook workbook
    c1, c2 = st.columns(2)
    with c1:
        if path.exists():
            st.download_button("⬇️ " + _t(lang, "Resultaten (Excel)", "Results (Excel)"),
                               data=path.read_bytes(), file_name=path.name,
                               mime=_xlsx_mime, width="stretch")
    with c2:
        if cb.exists():
            st.download_button("⬇️ " + _t(lang, "Codeboek (Excel)", "Codebook (Excel)"),
                               data=cb.read_bytes(), file_name=cb.name,
                               mime=_xlsx_mime, width="stretch")
    if path.exists():
        try:
            import pandas as pd
            # mixed-type columns → str so Arrow can render the preview
            df = pd.read_excel(path).astype(str)
            st.caption(f"{len(df)} {_t(lang, 'rijen', 'rows')} · "
                       f"{len(df.columns)} {_t(lang, 'kolommen', 'columns')}")
            st.dataframe(df.head(50), width="stretch", hide_index=True)
        except Exception as exc:
            st.caption(_t(lang, f"Voorbeeld niet beschikbaar: {exc}",
                          f"Preview unavailable: {exc}"))


# =============================================================================
# THE REGISTRY — one entry per pipeline step
# =============================================================================

STEP_VIEWS: Dict[int, StepView] = {
    0: StepView(),
    1: StepView(phases=("spell_check",)),
    2: StepView(phases=("quality_filter",),
                stats=stats_quality_filter),
    3: StepView(phases=("idea_extraction_context", "idea_extraction_taxonomy",
                        "idea_extraction_abstraction_ladder")),
    4: StepView(phases=tuple(f"classifier_p{i}" for i in range(1, 9))),
    5: StepView(phases=("codegen_p8", "codegen_p9"),
                stats=stats_codebook),
    6: StepView(phases=("code_assignment",),
                stats=stats_assignments,
                samples=samples_assignments_qa),
    7: StepView(stats=stats_export),
}
