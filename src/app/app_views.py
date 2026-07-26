"""
app_views.py — per-step view registry for the Streamlit app.

Each step contributes a StepView: which LLM phases it runs (for the read-only
model line on the RUN screen), and how to render its evidence on the OUTPUT
screen (stats panel, samples panel). Adding or enriching a step's presentation
is a registry entry here — page logic in app.py never changes for it.
(Design: app/dev/app_development_plan.md §3.4; HITL `review` slot is wired
for Phase D and unused until then.)

Loader results are cached with @st.cache_data, keyed on the dataset identity
plus an `epoch` that app.py bumps after every run/invalidation — so a re-run
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
def _raw(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_raw(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _preprocessed(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_preprocessed(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _extracted(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_extracted(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _extraction_meta(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_extraction_metadata(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _taxonomy(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_taxonomy(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _classified_ideas(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    # Reuse contract §3.6b: the step-4 console view's derivation layer, with
    # EXPLICIT dataset args (its defaults are import-time TEST_DATA bindings).
    from pipeline.step_4_classifier.view_assignments_facets import load_ideas
    try:
        return load_ideas(filename=filename, variable=var_name, sample_size=sample_size)
    except FileNotFoundError:
        return None


@st.cache_data(max_entries=16, show_spinner=False)
def _codebook(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_codebook(_spec(filename, var_name, sample_size))


@st.cache_data(max_entries=16, show_spinner=False)
def _assignments(filename: str, var_name: str, sample_size: Optional[int], epoch: int):
    return be.load_assignments(_spec(filename, var_name, sample_size))


# =============================================================================
# COSTS (Phase B6) — read-only view on the costs JSON (contract: plan §3.6c).
# The per-step date is always shown so a stale entry is recognizable.
# =============================================================================

def _usd(cost: float) -> str:
    return f"${cost:.2f}" if cost >= 0.01 else f"${cost:.4f}"


def render_cost_line(spec: DatasetSpec, lang: str, step: int):
    """One caption line on the OUTPUT screen: what this step's last run cost."""
    entry = be.step_costs(spec, step)
    tot = (entry or {}).get("total") or {}
    if tot.get("cost_usd") is None:
        return
    models = ", ".join(dict.fromkeys((entry.get("model_config") or {}).values()))
    bits = [_usd(tot["cost_usd"]),
            f"{tot.get('calls', 0)} {_t(lang, 'calls', 'calls')}"]
    if models:
        bits.append(models)
    if entry.get("date"):
        bits.append(entry["date"])
    st.caption(_t(lang, "Kosten", "Costs") + ": " + " · ".join(bits))


def costs_overview(spec: DatasetSpec, lang: str):
    """Run-total panel at Export: per-step costs + sum, dates always visible."""
    data = be.load_costs(spec)
    steps = (data or {}).get("steps", {})
    if not steps:
        return
    st.subheader(_t(lang, "Kosten van deze run", "Run costs"))
    rows, total = [], 0.0
    for n, key in be.STEP_COSTS_KEY.items():
        entry = steps.get(key)
        if not entry:
            continue
        tot = entry.get("total") or {}
        cost = tot.get("cost_usd", 0.0)
        total += cost
        rows.append({
            _t(lang, "stap", "step"): f"{n}. {be.STEP_LABELS[n]}",
            "model": ", ".join(dict.fromkeys((entry.get("model_config") or {}).values())),
            "calls": tot.get("calls", 0),
            "tokens": tot.get("input_tokens", 0) + tot.get("output_tokens", 0),
            "USD": f"{cost:.4f}",
            _t(lang, "datum", "date"): entry.get("date", ""),
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    dep = (data or {}).get("deployment") or {}
    dep_txt = " · ".join(v for v in (dep.get("provider"), dep.get("model_family")) if v)
    st.caption(f"**{_t(lang, 'Totaal', 'Total')}: {_usd(total)}**"
               + (f" · {dep_txt}" if dep_txt else ""))


# =============================================================================
# SAMPLING HELPER — k random items with a 🎲 re-roll button. The seed lives in
# session state, so unrelated reruns keep the same sample; only the button rolls.
# =============================================================================

def _rollable_sample(items: list, k: int, key: str, lang: str) -> list:
    import random
    if len(items) <= k:
        return list(items)
    if st.button("🎲 " + _t(lang, "Andere steekproef", "Another sample"),
                 key=f"{key}_roll"):
        st.session_state[key] = random.randrange(1 << 30)
    seed = st.session_state.setdefault(key, 0)
    return random.Random(seed).sample(list(items), k)


# =============================================================================
# STEP 1 — spell-check before/after (Phase B1)
# =============================================================================

def _formatting_key(text) -> str:
    """Casefold + strip punctuation AND whitespace: per the maintainer's
    definition capitals, punctuation and whitespace are layout, not correction
    ('Nvt' == 'N. v. t.')."""
    import re
    return re.sub(r"[\W_]+", "", str(text or "").casefold())


def _correction_pairs(spec: DatasetSpec, epoch: int):
    """[(raw, preprocessed)] joined on respondent_id, split into: blank (no
    answer → step 1's <NA> placeholder — missing-data handling, no correction),
    real word-level corrections, and formatting-only changes."""
    raw = _raw(spec.filename, spec.var_name, spec.sample_size, epoch)
    pre = _preprocessed(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not (raw and pre):
        return None, None, None, None
    pre_map = {p.respondent_id: p for p in pre}
    pairs, real, formatting, blank = [], [], [], []
    for r in raw:
        p = pre_map.get(r.respondent_id)
        if p is None:
            continue
        pairs.append((r, p))
        r_txt = str(r.response or "").strip()
        # Content-free raw ('' but also '?' or '.') or the NA placeholder as
        # outcome = missing-data handling, never a correction.
        if not _formatting_key(r_txt) or str(p.response or "").strip() == "<NA>":
            blank.append((r, p))
        elif r_txt != str(p.response or "").strip():
            if _formatting_key(r_txt) == _formatting_key(p.response):
                formatting.append((r, p))
            else:
                real.append((r, p))
    return pairs, real, formatting, blank


def stats_preprocessing(spec: DatasetSpec, lang: str, epoch: int):
    pairs, real, formatting, blank = _correction_pairs(spec, epoch)
    if pairs is None:
        return
    st.subheader(_t(lang, "Spellingscontrole", "Spell check"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(_t(lang, "Antwoorden", "Responses"), len(pairs))
    c2.metric(_t(lang, "Echt gecorrigeerd", "Really corrected"), len(real))
    c3.metric(_t(lang, "Alleen opmaak", "Formatting only"), len(formatting))
    c4.metric(_t(lang, "Ongewijzigd", "Unchanged"),
              len(pairs) - len(real) - len(formatting) - len(blank))
    if blank:
        st.caption(_t(lang, f"{len(blank)} leeg (→ NA-markering; stap 2 filtert ze)",
                      f"{len(blank)} blank (→ NA marker; step 2 filters them)"))


@st.fragment
def samples_preprocessing(spec: DatasetSpec, lang: str, epoch: int):
    _, real, formatting, _blank = _correction_pairs(spec, epoch)
    if real is None:
        return
    st.subheader(_t(lang, "Correcties — voor en na", "Corrections — before and after"))
    if not real:
        st.caption(_t(lang, "Geen inhoudelijke correcties — alleen opmaak "
                            "(hoofdletters/interpunctie).",
                      "No real corrections — formatting only (capitals/punctuation)."))
    for raw, pre in _rollable_sample(real, 5, f"s1_{spec.variable_key}", lang):
        with st.container(border=True):
            st.caption(f"`{raw.respondent_id}`")
            st.markdown(f"~~{raw.response}~~")
            st.markdown(f"**{pre.response}**")
    if formatting and st.toggle(
            _t(lang, f"Opmaakvoorbeelden ({len(formatting)})",
               f"Formatting examples ({len(formatting)})"),
            key=f"s1_fmt_{spec.variable_key}"):
        for raw, pre in formatting[:5]:
            st.markdown(f"- `{raw.respondent_id}` — {raw.response} → **{pre.response}**")


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


@st.fragment
def samples_quality_filter(spec: DatasetSpec, lang: str, epoch: int):
    """Excluded-response samples per category (Phase B2): the QA question is
    'did the filter throw away anything meaningful?' — so show what it excluded."""
    data = _quality_filtered(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not data:
        return
    labels = {99999997: _t(lang, "Weet niet / geen mening", "Don't know"),
              99999998: _t(lang, "Geen antwoord / leeg", "No answer / empty"),
              99999999: _t(lang, "Betekenisloos", "Gibberish")}
    excluded = {code: [d for d in data if getattr(d, "quality_filter_code", None) == code]
                for code in labels}
    if not any(excluded.values()):
        return
    st.subheader(_t(lang, "Uitgesloten antwoorden — steekproef",
                    "Excluded responses — sample"))
    for code, group in excluded.items():
        if not group:
            continue
        with st.expander(f"{labels[code]} ({len(group)})"):
            for d in _rollable_sample(group, 8, f"s2_{code}_{spec.variable_key}", lang):
                st.markdown(f"- `{d.respondent_id}` — {d.response}")


# =============================================================================
# STEP 3 — extraction lens + response → ideas samples (Phase B3)
# =============================================================================

def stats_extraction(spec: DatasetSpec, lang: str, epoch: int):
    meta = _extraction_meta(spec.filename, spec.var_name, spec.sample_size, epoch)
    data = _extracted(spec.filename, spec.var_name, spec.sample_size, epoch)
    if data:
        with_ideas = [m for m in data if getattr(m, "response_ideas", None)]
        n_ideas = sum(len(m.response_ideas) for m in with_ideas)
        st.subheader(_t(lang, "Idee-extractie", "Idea extraction"))
        c1, c2 = st.columns(2)
        c1.metric(_t(lang, "Ideeën", "Ideas"), n_ideas)
        c2.metric(_t(lang, "Antwoorden met ideeën", "Responses with ideas"), len(with_ideas))
    if not meta:
        return
    # The context lens: how the LLM was told to read this dataset (step 3, phase 1)
    st.markdown("**" + _t(lang, "Extractielens", "Extraction lens") + "**")
    lens = [(_t(lang, "taal", "language"), meta.lang), (_t(lang, "sector", "sector"), meta.sector),
            (_t(lang, "onderwerp", "topic"), meta.topic),
            (_t(lang, "perspectief", "perspective"), meta.perspective),
            (_t(lang, "entiteit", "entity"), meta.entity), (_t(lang, "intentie", "intent"), meta.intent)]
    st.caption(" · ".join(f"{k}: **{v}**" for k, v in lens if v))
    if meta.template_prefix:
        st.caption(_t(lang, "Sjabloon", "Template") + f": “{meta.template_prefix} …”")
    if meta.primary_dimension:
        st.markdown(f"**{_t(lang, 'Dimensie', 'Dimension')}:** {meta.primary_dimension}")
        if meta.primary_dimension_description:
            st.caption(meta.primary_dimension_description)
    if meta.domains:
        st.markdown("**" + _t(lang, "Domeinen", "Domains") + f"** ({len(meta.domains)})")
        st.dataframe([{_t(lang, "domein", "domain"): d.get("label", ""),
                       _t(lang, "definitie", "definition"): d.get("definition", "")}
                      for d in meta.domains],
                     width="stretch", hide_index=True)


@st.fragment
def samples_extraction(spec: DatasetSpec, lang: str, epoch: int):
    """One random response with its ideas: abstraction ladder + domain."""
    data = _extracted(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not data:
        return
    with_ideas = [m for m in data if getattr(m, "response_ideas", None)]
    if not with_ideas:
        return
    st.subheader(_t(lang, "Van antwoord naar ideeën", "From response to ideas"))
    m = _rollable_sample(with_ideas, 1, f"s3_{spec.variable_key}", lang)[0]
    st.markdown(f"**{_t(lang, 'Respondent', 'Respondent')}:** `{m.respondent_id}`")
    st.markdown(f"> {m.response}")
    for i in m.response_ideas:
        val = f" [{i.valence}]" if i.valence else ""
        dom = f" — _{i.domain}_" if i.domain else ""
        st.markdown(f"- **{i.instance}**{val}{dom}")
        if i.interpretation or i.abstraction:
            ladder = " → ".join(x for x in (i.interpretation, i.abstraction) if x)
            st.caption(f"&nbsp;&nbsp;&nbsp;↳ {ladder}")


# =============================================================================
# STEP 4 — taxonomy tree + structure health (Phase B4)
# =============================================================================

def stats_taxonomy(spec: DatasetSpec, lang: str, epoch: int):
    from collections import Counter
    ideas = _classified_ideas(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not ideas:
        return
    # Tree counts from the per-idea placements (same fields the console views use)
    tree = Counter((i.domain or "(?)", i.facet or "(?)", i.attribute or "(?)")
                   for i in ideas)
    domain_totals = Counter()
    for (dom, _, _), n in tree.items():
        domain_totals[dom] += n
    n_facets = len({(d, f) for d, f, _ in tree})
    st.subheader(_t(lang, "Taxonomie", "Taxonomy"))
    st.caption(f"{len(ideas)} {_t(lang, 'ideeën', 'ideas')} · "
               f"{len(domain_totals)} {_t(lang, 'domeinen', 'domains')} · "
               f"{n_facets} {_t(lang, 'facetten', 'facets')} · "
               f"{len(tree)} {_t(lang, 'attributen', 'attributes')}")
    for dom, dom_n in domain_totals.most_common():
        with st.expander(f"{dom} ({dom_n})"):
            rows = [{_t(lang, "facet", "facet"): f,
                     _t(lang, "attribuut", "attribute"): a, "n": n}
                    for (d, f, a), n in sorted(tree.items(), key=lambda kv: (kv[0][1], -kv[1]))
                    if d == dom]
            st.dataframe(rows, width="stretch", hide_index=True)

    # Structure health (read-only measure over the taxonomy cache)
    tax = _taxonomy(spec.filename, spec.var_name, spec.sample_size, epoch)
    if tax:
        from pipeline.step_4_classifier.taxonomy_health import measure
        with st.expander(_t(lang, "Structuurmeting", "Structure health")):
            st.code("\n".join(measure(tax).lines()), language=None)


# =============================================================================
# STEP 5 — codebook table
# =============================================================================

def stats_codebook(spec: DatasetSpec, lang: str, epoch: int):
    from collections import Counter
    codes = _codebook(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not (codes and codes.raw_codes):
        return
    st.subheader(_t(lang, "Codeboek", "Codebook") + f" ({len(codes.raw_codes)})")
    val = Counter(c.get("valence", "") for c in codes.raw_codes)
    st.caption(" · ".join(f"{v or '?'}: {n}" for v, n in val.most_common()))
    rows = [{"id": c.get("code_id", ""), "code": c.get("code_name", ""),
             "valence": c.get("valence", ""), "definition": c.get("definition", ""),
             "test": c.get("diagnostic_test", "")} for c in codes.raw_codes]
    st.dataframe(rows, width="stretch", hide_index=True)


# =============================================================================
# STEP 6 — code frequencies + respondent QA drill-down
# =============================================================================

def _code_id_to_name(spec: DatasetSpec, epoch: int) -> Dict[str, str]:
    """K# id → CURRENT codebook name, so a rename wins over the stored name."""
    codes = _codebook(spec.filename, spec.var_name, spec.sample_size, epoch)
    return {c["code_id"]: c["code_name"]
            for c in (codes.raw_codes if codes else []) if c.get("code_id")}


def stats_assignments(spec: DatasetSpec, lang: str, epoch: int):
    from collections import Counter
    models = _assignments(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not models:
        return
    # Reuse contract §3.6b: the console view's partition grouping. Display labels
    # go through assigned_code_id (K#) so the codebook's CURRENT name wins after
    # a rename — two stale names that map to one new name merge in the count.
    from pipeline.step_6_codeAssigner.view_assignments_codes import group_by_partition
    id_to_name = _code_id_to_name(spec, epoch)
    grouped = group_by_partition(models)
    counter = Counter()          # (partition, label) -> n
    unassigned = 0
    for part, by_code in grouped.items():
        for name, ideas in by_code.items():
            if name == "(unassigned)":
                unassigned += len(ideas)
                continue
            label = id_to_name.get(ideas[0].assigned_code_id) or name
            counter[(part, label)] += len(ideas)
    total = sum(counter.values()) or 1
    part_totals = Counter()
    for (part, _), n in counter.items():
        part_totals[part] += n
    st.subheader(_t(lang, "Codefrequenties", "Code frequencies"))
    st.caption(f"{total} {_t(lang, 'toegewezen ideeën', 'assigned ideas')}"
               + (f" · {unassigned} {_t(lang, 'niet toegewezen', 'unassigned')}"
                  if unassigned else ""))
    rows = [{_t(lang, "partitie", "partition"): part, "code": label,
             "n": n, "%": f"{100 * n / total:.1f}"}
            for (part, label), n in sorted(
                counter.items(), key=lambda kv: (-part_totals[kv[0][0]], kv[0][0], -kv[1]))]
    st.dataframe(rows, width="stretch", hide_index=True)


@st.fragment
def samples_assignments_qa(spec: DatasetSpec, lang: str, epoch: int):
    """QA drill-down: one respondent's assignments + rationale ("is this right?")."""
    models = _assignments(spec.filename, spec.var_name, spec.sample_size, epoch)
    if not models:
        return
    coded = [m for m in models if any(i.assigned_code for i in (m.response_ideas or []))]
    if not coded:
        return
    st.subheader(_t(lang, "Inspecteer een respondent", "Inspect a respondent"))
    id_to_name = _code_id_to_name(spec, epoch)
    m = _rollable_sample(coded, 1, f"qa6_{spec.variable_key}", lang)[0]
    st.markdown(f"**{_t(lang, 'Respondent', 'Respondent')}:** `{m.respondent_id}`")
    st.markdown(f"> {m.response}")
    for i in (m.response_ideas or []):
        if not i.assigned_code:
            continue
        label = id_to_name.get(i.assigned_code_id) or i.assigned_code
        conf = f" · {i.confidence:.2f}" if i.confidence is not None else ""
        st.markdown(f"- **{label}**{conf} — _{i.idea or i.instance}_")
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
            st.download_button(_t(lang, "Resultaten (Excel)", "Results (Excel)"),
                               data=path.read_bytes(), file_name=path.name,
                               mime=_xlsx_mime, width="stretch")
    with c2:
        if cb.exists():
            st.download_button(_t(lang, "Codeboek (Excel)", "Codebook (Excel)"),
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
    st.divider()
    costs_overview(spec, lang)


# =============================================================================
# THE REGISTRY — one entry per pipeline step
# =============================================================================

STEP_VIEWS: Dict[int, StepView] = {
    0: StepView(),
    1: StepView(phases=("spell_check",),
                stats=stats_preprocessing,
                samples=samples_preprocessing),
    2: StepView(phases=("quality_filter",),
                stats=stats_quality_filter,
                samples=samples_quality_filter),
    3: StepView(phases=("idea_extraction_context", "idea_extraction_taxonomy",
                        "idea_extraction_abstraction_ladder"),
                stats=stats_extraction,
                samples=samples_extraction),
    4: StepView(phases=tuple(f"classifier_p{i}" for i in range(1, 9)),
                stats=stats_taxonomy),
    5: StepView(phases=("codegen_p8", "codegen_p9"),
                stats=stats_codebook),
    6: StepView(phases=("code_assignment",),
                stats=stats_assignments,
                samples=samples_assignments_qa),
    7: StepView(stats=stats_export),
}
