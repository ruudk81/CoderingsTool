"""
Codebook verification layer (post-generation).

Pure Python, deterministic. Runs after the codebook is written (and the Overig
sweep) against the consolidated codebook and the step-4 taxonomy. Does NOT touch
generation logic — it only reports, so a run can be judged against a hard
definition of done.

Definition of done (PASS):
  - idea coverage = 100%        (every answer lands in a code)
  - attribute coverage = 100%   (every attribute is used by ≥1 code)
  - every code has ≥1 attribute (Overig exempt — always emitted, may be empty)
  - Overig ≤ 10% of ideas       (the catch-all stays small — the bare Overig
    code AND the codes nested under it, see `build_scorecard`)
  - hierarchy is intact         (no code the chain calls a child without a
    parent, no parent_code_id pointing at a code that does not exist)
  - no invalid sources          (provenance is real)
  - no taxonomy-level overlap   (no two same-valence codes with an identical
    source set)

All code↔taxonomy joins run on the attribute's stable id (A#) when the artifact
carries one, else on the name (naam-als-identiteit: renaming an attribute flips
no check). Pre-id codebooks translate source names to keys; a duplicated name
maps to all its ids — the same legacy tolerance as identity.ensure_codebook_ids.
One name link remains BY DESIGN: `attribute_assignments` (idea → attribute name)
shares its artifact with the attribute dicts, so a rename must update both in
the same edit — the HITL editor's contract, not a join this verifier can bridge.

Also reported (warnings, do NOT block PASS):
  - under-split codes: a code with a well-represented opposing pole that no
    counter-valence code sources — the pole has no home, so the code should
    likely have been split. Opposing ideas in attributes that a counter-valence
    code ALSO sources are not counted: they flow to that partner in step 6, so
    a valence-split pair does not flag itself.
  - overlap classes: benign valence split / partial overlap / taxonomy-level.
  - mini codes: a code whose expected idea volume (the matching pole of its
    source attributes) stays below the population floor — the parsimony
    counterweight to under-split, so over-differentiation is as visible as
    over-collapse.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

# Prevalence gate for the advisory checks below (under-split, mini codes).
# No % share requirement — population-scaled absolute floor only.


def _pole_clears(count: int, total: int) -> bool:
    """A valence pole is well-represented: population-scaled floor(log(n)) only."""
    if total <= 0:
        return False
    min_count = max(2, int(math.log(total)))
    return count >= min_count


# Welke valentiepolen een code van die valentie bezit. Alle vier de waarden
# van `ConsolidatedCode.valence` staan erin; alleen een ONBEKENDE waarde valt
# terug op het hele attribuut, want dan is er niets bekend om op te snijden.
_POLES = {
    "positive": ("positive",),
    "negative": ("negative",),
    "non_negative": ("positive", "neutral"),
    "neutral": ("neutral",),
}
_ALLE_POLEN = ("positive", "neutral", "negative")


def _pole_ideas(valence: str, counts: Dict[str, int]) -> int:
    """De ideeën van één attribuut die bij de POOL van een code horen — strikt.

    Voor de KINDEREN onder Overig. Een kind bezit niet zijn bronattribuut maar
    één pool ervan: hetzelfde attribuut voedt in de regel ook een hoofdcode.
    Het attribuut zelf tellen is geen benadering maar een andere grootheid — op
    set 7 gaf dat voor de kinderen 55,5% van alle ideeën tegen 3,6%
    valentiebewust, omdat 12 van hun 17 bronattributen ook een hoofdcode voeden.

    Dat geldt óók voor een NEUTRAAL kind, en die uitzondering is duur betaald:
    zolang `neutral` hier het hele attribuut claimde, viel `POLES="three"` (een
    instelling die de runner aanbiedt, en die neutrale polen oplevert die kind
    kunnen worden) op set 7 uit op 22,6% en FAIL, tegen 6,2% en PASS met de
    strikte pool. De redenering "neutraal sluit geen richting uit" gaat over
    Overig de OUDER — en die loopt hier niet langs: hij wordt op zijn
    bronattributen geteld, wat klopt omdat de sweep hem alleen attributen geeft
    die geen enkele code noemt.
    """
    return sum(counts.get(p, 0) for p in _POLES.get(valence, _ALLE_POLEN))


def _expected_ideas(valence: str, counts: Dict[str, int]) -> int:
    """Wat een code naar verwachting uit één attribuut draagt.

    Voor de MINI-CODEWAARSCHUWING, en daar wijkt precies één valentie af van
    `_pole_ideas`. Een neutrale HOOFDcode is per `models.py` dimensioneel — geen
    pool haalde de poort — en dekt zijn attribuut dus ongeacht richting; strikt
    op zijn neutrale pool tellen zou hem als over-gedifferentieerd melden
    terwijl hij het hele onderwerp draagt. Bij een kind is dat andersom, want
    daar staan de gerichte codes ernaast; vandaar twee functies en niet één.
    """
    if valence == "neutral":
        return sum(counts.get(p, 0) for p in _ALLE_POLEN)
    return _pole_ideas(valence, counts)


# =============================================================================
# SCORECARD MODELS
# =============================================================================

class CodePair(BaseModel):
    """A pair of codes that share one or more source attributes."""
    code_a: str
    code_b: str
    shared_attributes: List[str]


class AttributeOverlap(BaseModel):
    """A single attribute that feeds more than one code."""
    attribute: str
    code_names: List[str]
    valences: List[str]
    benign_valence_split: bool  # True when the codes span different valences


class UnderSplitCode(BaseModel):
    """A code with a well-represented, homeless opposing pole (true totals shown)."""
    code_name: str
    positive: int
    neutral: int
    negative: int


class MiniCode(BaseModel):
    """A code whose expected idea volume is below the population floor."""
    code_name: str
    valence: str
    expected_ideas: int


class CodebookScorecard(BaseModel):
    """Deterministic quality scorecard for a consolidated codebook."""
    n_codes: int
    n_attributes_total: int

    # Coverage
    attribute_coverage_pct: float
    idea_coverage_pct: float
    orphan_attributes: List[str] = Field(default_factory=list)
    orphan_idea_count: int = 0
    assigned_idea_count: int = 0

    # Catch-all. `overig_idea_share_pct` is de som van de twee helften
    # eronder — de poort telt ze allebei, en beide staan er los bij zodat een
    # lezer ononderscheiden materiaal van benoemde kinderen kan onderscheiden.
    overig_code_name: Optional[str] = None
    overig_idea_share_pct: float = 0.0
    overig_parent_idea_count: int = 0
    overig_child_idea_count: int = 0
    overig_child_code_names: List[str] = Field(default_factory=list)

    # Structural
    codes_without_attributes: List[str] = Field(default_factory=list)
    unknown_source_names: List[str] = Field(default_factory=list)
    children_without_parent: List[str] = Field(default_factory=list)
    dangling_parent_refs: List[str] = Field(default_factory=list)

    # Overlap (Mutual Exclusivity)
    overlap_attributes: List[AttributeOverlap] = Field(default_factory=list)
    taxonomy_level_pairs: List[CodePair] = Field(default_factory=list)
    partial_overlap_pairs: List[CodePair] = Field(default_factory=list)

    # Valence quality (warning, does not block PASS)
    under_split_codes: List[UnderSplitCode] = Field(default_factory=list)

    # Parsimony (warning, does not block PASS)
    mini_codes: List[MiniCode] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Hard definition of done."""
        return (
            self.attribute_coverage_pct == 100.0
            and self.idea_coverage_pct == 100.0
            and not self.codes_without_attributes
            and not self.unknown_source_names
            and not self.children_without_parent
            and not self.dangling_parent_refs
            and self.overig_idea_share_pct <= 10.0
            and not self.taxonomy_level_pairs
        )

    def failure_reasons(self) -> List[str]:
        reasons = []
        if self.idea_coverage_pct < 100.0:
            reasons.append(f"idea coverage {self.idea_coverage_pct}% < 100%")
        if self.attribute_coverage_pct < 100.0:
            reasons.append(f"attribute coverage {self.attribute_coverage_pct}% < 100%")
        if self.codes_without_attributes:
            reasons.append(f"{len(self.codes_without_attributes)} code(s) without attributes")
        if self.unknown_source_names:
            reasons.append(f"{len(self.unknown_source_names)} invalid source name(s)")
        if self.children_without_parent:
            reasons.append(f"{len(self.children_without_parent)} child code(s) without a parent")
        if self.dangling_parent_refs:
            reasons.append(f"{len(self.dangling_parent_refs)} dangling parent reference(s)")
        if self.overig_idea_share_pct > 10.0:
            reasons.append(f"Overig {self.overig_idea_share_pct}% > 10%")
        if self.taxonomy_level_pairs:
            reasons.append(f"{len(self.taxonomy_level_pairs)} taxonomy-level overlap pair(s)")
        return reasons


# =============================================================================
# TAXONOMY EXTRACTION
# =============================================================================

def _attr(obj: Any, name: str) -> Any:
    """Read a field from a Pydantic model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def collect_taxonomy_attributes(partition_results: Dict[str, Any]) -> List[str]:
    """Ground-truth attribute names: union of attribute_name across all domains."""
    names: List[str] = []
    seen = set()
    for result in partition_results.values():
        attributes = _attr(result, "attributes") or {}
        for attr_list in attributes.values():
            for attr in attr_list:
                name = _attr(attr, "attribute_name")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def collect_idea_assignments(partition_results: Dict[str, Any]) -> Dict[str, str]:
    """Map idea_id -> attribute_name across all domains."""
    assignments: Dict[str, str] = {}
    for result in partition_results.values():
        assignments.update(_attr(result, "attribute_assignments") or {})
    return assignments


def collect_attribute_valence(partition_results: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per-attribute idea valence breakdown: attr -> {positive, neutral, negative}.

    Aggregates step-4's per-idea valence (no valence → neutral)."""
    counts: Dict[str, Dict[str, int]] = {}
    for result in partition_results.values():
        assign = _attr(result, "attribute_assignments") or {}
        valence = _attr(result, "attribute_valence") or {}
        for idea_id, attr in assign.items():
            v = valence.get(idea_id)
            bucket = "positive" if v == "+" else "negative" if v == "-" else "neutral"
            counts.setdefault(attr, {"positive": 0, "neutral": 0, "negative": 0})[bucket] += 1
    return counts


# =============================================================================
# SCORECARD BUILDER
# =============================================================================

def build_scorecard(
    codes: List[Any],
    partition_results: Dict[str, Any],
    overig_code_name: Optional[str] = None,
    child_code_ids: Optional[Set[str]] = None,
) -> CodebookScorecard:
    """Build a deterministic scorecard for a codebook against its taxonomy.

    Args:
        codes: List of ConsolidatedCode (or dicts) with code_name, valence, source_attributes
        partition_results: step-4 partition_results (DomainResultModel objects or dicts)
        overig_code_name: name of the catch-all code, if one was added by the Overig sweep
        child_code_ids: de `K#`'s die de keten als KIND bedoelde (uit de vormen,
            `origin == "child"`), gelegd naast het `parent_code_id`-veld op de
            code zelf. Lopen ze uiteen, dan telt een kind stil als hoofdcode mee
            en valt het buiten het Overig-plafond.

            EXPLICIET EEN STRUIKELDRAAD, geen dekking: op geen enkel pad dat
            vandaag bestaat kunnen de twee verschillen. `models.py` negeert een
            verkeerd gespelde init-KWARG stilzwijgend
            (`ConsolidatedCode(parent_code=...)` → `parent_code_id is None`),
            maar een verkeerd gespelde ATTRIBUUTtoekenning is luid
            (`code.parent_code = ...` → ValueError). `link_children_to_overig`
            doet het tweede, dus die kan het defect niet maken; en een code die
            elders met de foute kwarg gebouwd wordt, staat om dezelfde reden ook
            niet in deze lijst. Een tweede afleiding uit de vormen
            (`_shape_lookup`/`_match_shape`) helpt niet: dát is de afleiding die
            `result.shapes` al opleverde, en `codebook_writer` bouwt elke code's
            bronnamen en valentie ÚIT zijn vorm, dus hermatchen geeft per
            constructie dezelfde vorm terug. Een afleiding die het niet oneens
            kan zijn is geen tweede mening; er een kopen zou een tweede bron van
            waarheid voor kindschap betekenen, en de hiërarchie hoort in één
            veld te leven.

            Wat hij wél vangt, en dat is niet niets: een kind dat zijn ouder
            KWIJTRAAKT tussen koppelen en beoordelen — een latere mutatie, een
            herbouwde lijst, een cache-rondgang die het veld laat vallen, of een
            toekomstig bouwpad dat de ouder zelf zet. Kosten: één
            verzamelingsvergelijking.
    """
    idea_assignments = collect_idea_assignments(partition_results)
    attr_valence = collect_attribute_valence(partition_results)

    # --- Key space -----------------------------------------------------------
    # Every join below runs on an attribute KEY: its stable id (A#) when the
    # artifact carries one, else its name. Codes with source_attribute_ids join
    # positionally by id, so a renamed attribute keeps its coverage and
    # provenance; pre-id codes translate names to keys (a duplicated name maps
    # to ALL its ids — the ensure_codebook_ids legacy tolerance, under which
    # id/name list lengths can differ and the code falls back to name keys).
    attr_keys: List[str] = []                 # universe, taxonomy order
    key_to_name: Dict[str, str] = {}          # key -> current display name
    name_to_keys: Dict[str, List[str]] = {}
    for result in partition_results.values():
        for attr_list in (_attr(result, "attributes") or {}).values():
            for attr in attr_list:
                name = _attr(attr, "attribute_name")
                if not name:
                    continue
                akey = _attr(attr, "attribute_id") or name
                if akey not in key_to_name:
                    attr_keys.append(akey)
                    key_to_name[akey] = name
                keys = name_to_keys.setdefault(name, [])
                if akey not in keys:
                    keys.append(akey)
    key_set = set(attr_keys)

    def source_keys(code) -> List[tuple]:
        """[(key, source_name)] per code source."""
        sources = _attr(code, "source_attributes") or []
        ids = _attr(code, "source_attribute_ids") or []
        if ids and len(ids) == len(sources):
            return list(zip(ids, sources))
        return [(k, s) for s in sources for k in (name_to_keys.get(s) or [s])]

    def idea_keys(attr_name: str) -> List[str]:
        return name_to_keys.get(attr_name) or [attr_name]

    # Legitimate source keys: taxonomy attributes OR attributes ideas were
    # assigned to (the latter lets Overig absorb step-4 dangling assignments
    # without tripping "unknown"; dangling names key as themselves).
    legitimate = key_set | set(idea_assignments.values())

    # key -> [(code_name, valence), ...]
    attr_to_codes: Dict[str, List[tuple]] = {}
    unknown: List[str] = []
    unknown_seen = set()
    codes_without_attributes: List[str] = []
    for code in codes:
        code_name = _attr(code, "code_name") or ""
        valence = _attr(code, "valence") or ""
        pairs = source_keys(code)
        if not pairs and code_name != overig_code_name:
            # Overig is always emitted and may legitimately be empty (no
            # dangling assignments, no zero-embedding/zero-assignment
            # attributes to route) — that's a valid, passing state, not a
            # structural defect. Any other sourceless code still FAILs.
            codes_without_attributes.append(code_name)
        for skey, sname in pairs:
            attr_to_codes.setdefault(skey, []).append((code_name, valence))
            if skey not in legitimate and sname not in legitimate and sname not in unknown_seen:
                unknown_seen.add(sname)
                unknown.append(sname)

    covered_any = set(attr_to_codes)                          # any code source (incl. Overig)

    # --- Coverage ---
    orphans = [key_to_name[k] for k in attr_keys if k not in covered_any]
    attr_coverage = (sum(1 for k in attr_keys if k in covered_any) / len(attr_keys)) if attr_keys else 1.0

    assigned = len(idea_assignments)
    covered_ideas = sum(1 for attr in idea_assignments.values()
                        if any(k in covered_any for k in idea_keys(attr)))
    idea_coverage = (covered_ideas / assigned) if assigned else 1.0

    # --- Overig share: de kale ouder ÉN de codes die eronder hangen ---
    # Het plafond bewaakt hoeveel materiaal het HOOFDcodeboek niet plaatste, en
    # een kind staat daar net zo goed buiten als de ouder. Zou de poort alleen
    # de kale ouder tellen, dan verlaagt elke verhuizing van Overig naar een
    # kind het cijfer zonder dat er iets beter geplaatst is — het plafond wordt
    # dan omzeilbaar door te nestelen. Op set 7 leest de ouder alleen 0,2% en
    # ouder plus kinderen 3,8%; de ouder hield vóór de kinderen 99 respondenten
    # en houdt er nu 8, dus de eerste lezing meet die verhuizing weg terwijl er
    # niets beter geplaatst is.
    #
    # Het tegenargument is echt en wordt hier NIET met het totaal beantwoord
    # maar met de rapportage: een kind is geen ononderscheiden materiaal, het
    # draagt een naam, een richting en een facet. Daarom staan de twee helften
    # er altijd los bij — wie het cijfer ziet naderen kan zien welke helft het
    # drijft, en dat is precies wat één samengesteld getal niet kon.
    #
    # De twee helften mogen opgeteld worden omdat ze per constructie disjunct
    # zijn: de sweep vult Overig met attributen die GEEN code noemt, en een
    # kind is een code.
    overig_share = 0.0
    parent_ideas = 0
    child_ideas = 0
    child_names: List[str] = []
    if overig_code_name:
        overig_code = next((c for c in codes
                            if (_attr(c, "code_name") or "") == overig_code_name), None)
        overig_keys = ({k for k, _ in source_keys(overig_code)}
                       if overig_code is not None else set())
        overig_id = (_attr(overig_code, "code_id") or "") if overig_code is not None else ""
        parent_ideas = sum(1 for attr in idea_assignments.values()
                           if any(k in overig_keys for k in idea_keys(attr)))
        # `overig_id` leeg betekent niet stil terugvallen op de zwakkere regel:
        # `mint_code_ids` mint het hele boek of niets, dus zonder id op Overig
        # draagt geen enkele code een ouder (nul kinderen is dan het juiste
        # antwoord) óf wijst een ouder naar een id dat niet bestaat — en dat
        # meldt `dangling_parent_refs` hieronder als defect dat de poort faalt.
        for code in codes:
            if overig_id and (_attr(code, "parent_code_id") or "") == overig_id:
                child_names.append(_attr(code, "code_name") or "")
                child_ideas += sum(
                    _pole_ideas(_attr(code, "valence") or "", attr_valence.get(a, {}))
                    for a in (_attr(code, "source_attributes") or []))
        overig_share = ((parent_ideas + child_ideas) / assigned * 100) if assigned else 0.0

    # --- Hiërarchie: de bedoeling tegen het veld ---
    all_code_ids = {_attr(c, "code_id") or "" for c in codes}
    bedoelde_kinderen = child_code_ids or set()
    children_without_parent: List[str] = []
    dangling_parent_refs: List[str] = []
    for code in codes:
        name = _attr(code, "code_name") or ""
        parent = _attr(code, "parent_code_id") or ""
        code_id = _attr(code, "code_id") or ""
        if code_id and code_id in bedoelde_kinderen and not parent:
            children_without_parent.append(name)
        if parent and parent not in all_code_ids:
            dangling_parent_refs.append(name)

    # --- Attribute-level overlap ---
    overlaps: List[AttributeOverlap] = []
    for skey, pairs in attr_to_codes.items():
        if len(pairs) < 2 or skey not in key_set:
            continue
        valences = [p[1] for p in pairs]
        overlaps.append(AttributeOverlap(
            attribute=key_to_name.get(skey, skey),
            code_names=[p[0] for p in pairs],
            valences=valences,
            benign_valence_split=len(set(valences)) > 1,
        ))

    # --- Code-pair overlap (taxonomy-level vs partial overlap) ---
    # Same key space: ids keep same-named attributes in different domains
    # distinct — name-sets would falsely collide them.
    code_sources = []  # (code_name, valence, set(source keys ∩ taxonomy))
    for code in codes:
        srcs = {k for k, _ in source_keys(code) if k in key_set}
        code_sources.append((_attr(code, "code_name") or "", _attr(code, "valence") or "", srcs))

    taxonomy_pairs: List[CodePair] = []
    review_pairs: List[CodePair] = []
    for (name_a, val_a, src_a), (name_b, val_b, src_b) in combinations(code_sources, 2):
        shared = src_a & src_b
        if not shared or val_a != val_b:
            continue
        if src_a == src_b:
            taxonomy_pairs.append(CodePair(code_a=name_a, code_b=name_b, shared_attributes=sorted(shared)))
        else:
            review_pairs.append(CodePair(code_a=name_a, code_b=name_b, shared_attributes=sorted(shared)))

    # --- Under-split detection (over-collapse) ---
    # Advisory only — reported in the scorecard, does not block PASS.
    # The gate counts only HOMELESS opposing ideas: opposing ideas in an
    # attribute that a counter-valence code also sources flow to that partner
    # in step 6, so a valence-split pair must not flag itself. Reported counts
    # stay the true attribute totals.
    attr_code_valences: Dict[str, set] = {}
    for code in codes:
        v = _attr(code, "valence") or ""
        for attr in (_attr(code, "source_attributes") or []):
            attr_code_valences.setdefault(attr, set()).add(v)

    under_split: List[UnderSplitCode] = []
    for code in codes:
        code_valence = _attr(code, "valence") or ""
        if (_attr(code, "code_name") or "") == overig_code_name:
            continue
        p = n = g = 0
        homeless_p = homeless_g = 0
        for attr in (_attr(code, "source_attributes") or []):
            c = attr_valence.get(attr, {})
            ap, ag = c.get("positive", 0), c.get("negative", 0)
            p += ap
            n += c.get("neutral", 0)
            g += ag
            covering = attr_code_valences.get(attr, set())
            if "positive" not in covering:
                homeless_p += ap
            if "negative" not in covering:
                homeless_g += ag
        total = p + n + g
        # Flag whenever an OPPOSING pole is well-represented AND homeless.
        if code_valence == "neutral":
            flagged = _pole_clears(homeless_p, total) and _pole_clears(homeless_g, total)
        elif code_valence == "positive":
            flagged = _pole_clears(homeless_g, total)
        elif code_valence == "negative":
            flagged = _pole_clears(homeless_p, total)
        else:
            flagged = False
        if flagged:
            under_split.append(UnderSplitCode(
                code_name=_attr(code, "code_name") or "", positive=p, neutral=n, negative=g))

    # --- Mini-code detection (over-differentiation) ---
    # Advisory counterweight to under-split. Expected volume = the matching
    # pole of the code's source attributes (`_expected_ideas`). Een
    # `non_negative` code kreeg hier tot 2026-08-22 al zijn negatieve ideeën
    # meegeteld, waardoor hij nooit onder de bodem kon uitkomen: de
    # waarschuwing zweeg juist bij de codes die hem nodig hadden. Op set 7
    # verandert de gemelde LIJST niet, maar 25 van de 43 codes krijgen een
    # andere verwachting — de lijst is daar stabiel met marge, niet per
    # constructie. Onder de bevolkingsbodem floor(log(assigned)) kan een code
    # zichzelf waarschijnlijk niet dragen.
    mini_floor = max(2, int(math.log(assigned))) if assigned > 0 else 2
    mini_codes: List[MiniCode] = []
    for code in codes:
        code_name = _attr(code, "code_name") or ""
        if code_name == overig_code_name:
            continue
        code_valence = _attr(code, "valence") or ""
        expected = sum(_expected_ideas(code_valence, attr_valence.get(attr, {}))
                       for attr in (_attr(code, "source_attributes") or []))
        if expected < mini_floor:
            mini_codes.append(MiniCode(
                code_name=code_name, valence=code_valence, expected_ideas=expected))

    return CodebookScorecard(
        n_codes=len(codes),
        n_attributes_total=len(attr_keys),
        attribute_coverage_pct=round(attr_coverage * 100, 1),
        idea_coverage_pct=round(idea_coverage * 100, 1),
        orphan_attributes=orphans,
        orphan_idea_count=assigned - covered_ideas,
        assigned_idea_count=assigned,
        overig_code_name=overig_code_name,
        overig_idea_share_pct=round(overig_share, 1),
        overig_parent_idea_count=parent_ideas,
        overig_child_idea_count=child_ideas,
        overig_child_code_names=child_names,
        codes_without_attributes=codes_without_attributes,
        unknown_source_names=unknown,
        children_without_parent=children_without_parent,
        dangling_parent_refs=dangling_parent_refs,
        overlap_attributes=overlaps,
        taxonomy_level_pairs=taxonomy_pairs,
        partial_overlap_pairs=review_pairs,
        under_split_codes=under_split,
        mini_codes=mini_codes,
    )


# =============================================================================
# FORMATTING
# =============================================================================

def format_scorecard(sc: CodebookScorecard) -> str:
    """Render the scorecard as a human-readable console block."""
    status = "PASS" if sc.passed else "FAIL"
    lines = [
        "=" * 80,
        f"CODEBOOK SCORECARD — {status}",
        "=" * 80,
        f"  Codes:               {sc.n_codes}",
        f"  Attributes (taxon.): {sc.n_attributes_total}",
        f"  Attribute coverage:  {sc.attribute_coverage_pct}%",
        f"  Idea coverage:       {sc.idea_coverage_pct}%  "
        f"({sc.orphan_idea_count}/{sc.assigned_idea_count} uncovered)",
    ]
    if sc.overig_code_name:
        # Altijd beide helften, ook als er nul kinderen zijn: welke helft de
        # poort ook telt, een lezer moet ze los kunnen zien. Het totaal alleen
        # kan een ononderscheiden catch-all niet onderscheiden van evenveel
        # materiaal in benoemde, gerichte kinderen.
        lines.append(f"  Overig share:        {sc.overig_idea_share_pct}%  (cap 10%)"
                     f"  — parent + children")
        lines.append(f"      in '{sc.overig_code_name}' itself: "
                     f"{sc.overig_parent_idea_count} idea(s)")
        lines.append(f"      in {len(sc.overig_child_code_names)} child code(s): "
                     f"{sc.overig_child_idea_count} idea(s)")
        for naam in sc.overig_child_code_names:
            lines.append(f"          - {naam}")

    if not sc.passed:
        lines.append(f"\n  ✗ FAIL — {'; '.join(sc.failure_reasons())}")

    if sc.orphan_attributes:
        lines.append(f"\n  ⚠ ORPHAN ATTRIBUTES — covered by no code ({len(sc.orphan_attributes)}):")
        for a in sc.orphan_attributes:
            lines.append(f"      - {a}")

    if sc.codes_without_attributes:
        lines.append(f"\n  ⚠ CODES WITHOUT ATTRIBUTES ({len(sc.codes_without_attributes)}):")
        for c in sc.codes_without_attributes:
            lines.append(f"      - {c}")

    if sc.unknown_source_names:
        lines.append(f"\n  ⚠ UNKNOWN SOURCE NAMES ({len(sc.unknown_source_names)}):")
        for a in sc.unknown_source_names:
            lines.append(f"      - {a}")

    if sc.children_without_parent:
        lines.append(f"\n  ⚠ CHILD CODES WITHOUT A PARENT "
                     f"({len(sc.children_without_parent)}) — the chain built these as "
                     f"children but they carry no parent_code_id, so they count as "
                     f"ordinary top-level codes:")
        for c in sc.children_without_parent:
            lines.append(f"      - {c}")

    if sc.dangling_parent_refs:
        lines.append(f"\n  ⚠ DANGLING PARENT REFERENCES ({len(sc.dangling_parent_refs)}) — "
                     f"parent_code_id points at a code that does not exist:")
        for c in sc.dangling_parent_refs:
            lines.append(f"      - {c}")

    if sc.taxonomy_level_pairs:
        lines.append(f"\n  ⚠ TAXONOMY-LEVEL OVERLAP — identical source set "
                     f"({len(sc.taxonomy_level_pairs)}) — escalate to taxonomy:")
        for p in sc.taxonomy_level_pairs:
            lines.append(f"      - \"{p.code_a}\" / \"{p.code_b}\"  ←  {', '.join(p.shared_attributes)}")

    if sc.under_split_codes:
        lines.append(f"\n  ◦ UNDER-SPLIT (warning) — well-represented opposing pole without a "
                     f"counter-direction code ({len(sc.under_split_codes)}) — likely should be split:")
        for u in sc.under_split_codes:
            lines.append(f"      - \"{u.code_name}\"  (+{u.positive} / ○{u.neutral} / −{u.negative})")

    if sc.mini_codes:
        lines.append(f"\n  ◦ MINI-CODES (warning) — expected volume below floor(log(n)) "
                     f"({len(sc.mini_codes)}) — over-differentiation signal:")
        for m in sc.mini_codes:
            lines.append(f"      - \"{m.code_name}\" [{m.valence}]  (~{m.expected_ideas} ideas)")

    if sc.partial_overlap_pairs:
        lines.append(f"\n  ◦ PARTIAL OVERLAP — same-valence partial overlap "
                     f"({len(sc.partial_overlap_pairs)}):")
        for p in sc.partial_overlap_pairs:
            lines.append(f"      - \"{p.code_a}\" / \"{p.code_b}\"  ←  {', '.join(p.shared_attributes)}")

    lines.append("=" * 80)
    return "\n".join(lines)
