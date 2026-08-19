#%%

"""Meet hoe stabiel step 4's boom is van run tot run. Read-only, geen LLM-calls.

Draai dit na elke step-4-run. Het legt een snapshot vast en vergelijkt met elke
eerdere snapshot van dezelfde dataset+variabele, want een run overschrijft de
cache van de vorige — zonder een snapshot op dat moment is die vorige run weg.
Gemodelleerd op `step_3_ideaExtractor/measure_stability.py`, dat hetzelfde doet
een laag hoger.

De vraag waarvoor dit gebouwd is: step 5's consolidatie reproduceert niet, en het
alternatief op de tekentafel is een deterministische SNEDE door deze boom. Een
snede reproduceert per definitie — maar alleen ten opzichte van de boom. Wiebelt
de boom zelf, dan verhuist de variantie van step 5 naar step 4 in plaats van te
verdwijnen. Dat is wat hier gemeten wordt, vóór er iets gebouwd wordt.

Drie lagen, apart gemeten, want ze kunnen los van elkaar wiebelen:

  DOMEIN     de grofste indeling, grotendeels overgenomen uit step 3
  FACET      de laag waarop het snede-ontwerp zou snijden — dit is het getal
             dat de beslissing draagt
  ATTRIBUUT  de fijnste laag; hierop groepeert step 5 vandaag

Gemeten met de Adjusted Rand Index over IDEEËN, nooit over namen. ARI vraagt of
twee ideeën die in run A samen zaten dat in run B nog steeds doen, en heeft
daarvoor geen labelmatching nodig. Dat is hier geen luxe maar noodzaak: step 4
verzint zijn facet- en attribuutnamen elke run opnieuw en `identity.py` mint
verse ids, dus elke meting die op naam of id joint meet ruis.
1,0 = identieke indeling, 0,0 = niet beter dan toeval.

De eenheid is `idea_id`, gezet door step 3 en ongewijzigd zolang step 3 niet
opnieuw draait. Draait step 3 wél opnieuw, dan vergelijk je twee verschillende
ideeënverzamelingen en is de uitkomst betekenisloos; het script meldt dat zelf
zodra de overlap tussen snapshots niet volledig is.

LET OP — anders dan bij step 3 is hier GEEN gratis ruisvloer te halen. Step 3
beoordeelt elk antwoord apart, dus identieke tekst die daar op twee domeinen
landt is aantoonbaar ruis. Step 4 wijst toe per (domein, uniek label), waarbij
het label `instance + interpretation` is: identieke tekst binnen één domein
krijgt één call en wordt als blok verplaatst (`assignment_batching.py`). Twee
identieke labels in hetzelfde domein KUNNEN dus niet uiteenlopen — dat is een
invariant van de constructie, geen meting.

Wat dit script daarom in plaats daarvan doet:

  blokinvariant   controleert die aanname. (domein, label) -> precies één
                  attribuut. Nul overtredingen verwacht; is het niet nul, dan is
                  er een bug en zijn de ARI's beneden niet te vertrouwen.
  geërfde spreiding  dezelfde letterlijke tekstspan die tóch in verschillende
                  attributen landt, uitgesplitst naar oorzaak: een ander domein
                  (step 3's toewijzing) of een andere interpretatie (step 3's
                  bewoording). Dat is variantie die step 4 ERFT, niet maakt —
                  nuttig om te weten, maar het is geen bodem voor de ARI.

De bodem voor step 4 moet dus uit de herhaalde runs zelf komen; er is geen
gratis variant.

    python -m pipeline.step_4_classifier.measure_taxonomy_stability
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
import models

# Eén implementatie van ARI in de codebase, niet twee. De correctieterm is het
# hele punt van dat ding — een ruw overeenstemmingspercentage ziet er altijd hoog
# uit zodra één cluster domineert — en die wil je niet in twee versies hebben.
from pipeline.step_3_ideaExtractor.measure_stability import adjusted_rand_index

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

SNAPSHOT_FILE = project_root / "data" / "step4_stability.jsonl"

LEVELS = ("domein", "facet", "attribuut")

# De drempel waarop step 5 beslist of iets een eigen code mag zijn. Hier alleen
# gebruikt om te tellen hoeveel facetten er onder blijven — dat is de tweede
# falsificatiegrens van het snede-ontwerp (zie step 5's WORK.md).
T_KEEP_SHARE = 0.01
T_KEEP_MIN = 3


# =============================================================================
# MEASUREMENT
# =============================================================================

def _placement_by_attribute(taxonomy) -> Dict[str, Tuple[str, str]]:
    """attribuut-id -> (domein, facet), afgeleid uit de STRUCTUUR.

    Niet uit de velden op het idee zelf. Die twee kunnen uiteenlopen, en de
    structuur is de afgesproken enige bron (zie step 4's dev-docs over de remap);
    een idee dat zijn oude facetnaam nog meedraagt zou anders als een eigen facet
    meetellen en de meting optisch instabieler maken dan de boom is.
    """
    placement: Dict[str, Tuple[str, str]] = {}
    for domain_name, domain in (taxonomy.partition_results or {}).items():
        attributes = domain["attributes"] if isinstance(domain, dict) else domain.attributes
        for facet_name, attribute_list in (attributes or {}).items():
            for attribute in attribute_list or []:
                attribute_id = attribute.get("attribute_id", "")
                if attribute_id:
                    placement[attribute_id] = (domain_name, facet_name)
    return placement


def build_snapshot(classified, taxonomy) -> Dict:
    """Eén run, teruggebracht tot wat tegen een andere run te leggen is."""
    placement = _placement_by_attribute(taxonomy)

    labels: Dict[str, Dict[str, str]] = {level: {} for level in LEVELS}
    texts: Dict[str, str] = {}
    label_texts: Dict[str, str] = {}
    respondents_by_facet: Dict[str, set] = defaultdict(set)
    attributes_by_facet: Dict[str, set] = defaultdict(set)
    respondents = set()

    n_ideas_total = 0
    n_unplaced = 0

    for response in classified:
        respondent_id = str(response.respondent_id)
        respondents.add(respondent_id)
        for idea in response.response_ideas or []:
            n_ideas_total += 1
            attribute_id = idea.attribute_id or ""
            if not attribute_id or attribute_id not in placement:
                n_unplaced += 1
                continue
            domain, facet = placement[attribute_id]
            facet_label = f"{domain} > {facet}"

            labels["domein"][idea.idea_id] = domain
            labels["facet"][idea.idea_id] = facet_label
            labels["attribuut"][idea.idea_id] = attribute_id

            texts[idea.idea_id] = (idea.instance or idea.idea or "").strip().lower()
            # Het label waarop step 4 daadwerkelijk toewijst, niet de kale span.
            label_texts[idea.idea_id] = " > ".join(
                part for part in ((idea.instance or "").strip().lower(),
                                  (idea.interpretation or "").strip().lower())
                if part)
            respondents_by_facet[facet_label].add(respondent_id)
            attributes_by_facet[facet_label].add(attribute_id)

    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "filename": FILENAME,
        "variable": VARIABLE,
        "sample_size": SAMPLE_SIZE,
        "respondents": len(respondents),
        "ideas_total": n_ideas_total,
        "ideas_placed": len(labels["attribuut"]),
        "ideas_unplaced": n_unplaced,
        "n_domains": len(set(labels["domein"].values())),
        "n_facets": len(set(labels["facet"].values())),
        "n_attributes": len(set(labels["attribuut"].values())),
        "labels": labels,
        "texts": texts,
        "label_texts": label_texts,
        "facet_reach": {f: len(r) for f, r in respondents_by_facet.items()},
        "facet_attributes": {f: len(a) for f, a in attributes_by_facet.items()},
    }


def block_move_violations(snapshot: Dict) -> Dict:
    """Toetst de invariant waarop step 4's toewijzing rust.

    `assignment_batching.group_label_reps` groepeert per uniek label binnen een
    domein, dus (domein, label) hoort op precies één attribuut uit te komen. Nul
    overtredingen is de verwachting, niet de hoop. Komt hier iets anders uit, dan
    is er iets stuk in de blokverplaatsing en zijn de ARI's hieronder niet te
    duiden — dan meet je een bug, geen instabiliteit.
    """
    by_key = defaultdict(set)
    for idea_id, attribute in snapshot["labels"]["attribuut"].items():
        label = snapshot.get("label_texts", {}).get(idea_id, "")
        if label:
            by_key[(snapshot["labels"]["domein"][idea_id], label)].add(attribute)

    broken = {key: sorted(attrs) for key, attrs in by_key.items() if len(attrs) > 1}
    return {"keys": len(by_key), "violations": len(broken), "detail": broken}


def inherited_spread(snapshot: Dict) -> Dict:
    """Dezelfde letterlijke span die tóch op verschillende attributen landt.

    Geen ruisvloer van step 4 — zie de moduledocstring — maar variantie die uit
    step 3 binnenkomt. Uitgesplitst naar oorzaak, want de twee vragen om een
    andere ingreep: een span die over domeinen uiteenvalt is step 3's
    domeintoewijzing, een span die binnen één domein uiteenvalt is step 3's
    bewoording van `interpretation` die twee reps maakt van wat één ding is.
    """
    by_span = defaultdict(list)
    for idea_id, attribute in snapshot["labels"]["attribuut"].items():
        span = snapshot["texts"].get(idea_id, "")
        if span:
            by_span[span].append((snapshot["labels"]["domein"][idea_id], attribute))

    repeated = {s: v for s, v in by_span.items() if len(v) > 1}
    split = {s: v for s, v in repeated.items() if len({a for _d, a in v}) > 1}

    across_domains = {s: v for s, v in split.items() if len({d for d, _a in v}) > 1}
    within_domain = {s: v for s, v in split.items() if s not in across_domains}

    n_repeated = sum(len(v) for v in repeated.values())
    minority = sum(
        len(v) - Counter(a for _d, a in v).most_common(1)[0][1] for v in split.values()
    )

    return {
        "repeated_ideas": n_repeated,
        "repeated_spans": len(repeated),
        "split_spans": len(split),
        "across_domains": len(across_domains),
        "within_domain": len(within_domain),
        "minority": minority,
        "pct": round(100 * minority / n_repeated, 1) if n_repeated else 0.0,
        "detail": sorted(
            ((s, len({d for d, _a in v}), len({a for _d, a in v}), len(v))
             for s, v in split.items()),
            key=lambda row: -row[3]),
    }


def thin_facets(snapshot: Dict) -> Tuple[int, int]:
    """(facetten onder de drempel, totaal). De tweede falsificatiegrens van het
    snede-ontwerp: levert meer dan een kwart van de facetten geen eigen code op,
    dan is de facetlaag niet de juiste korrel om mee te beginnen."""
    threshold = max(T_KEEP_MIN, round(T_KEEP_SHARE * snapshot["respondents"]))
    reach = snapshot["facet_reach"]
    return sum(1 for n in reach.values() if n < threshold), len(reach)


# =============================================================================
# REPORTING
# =============================================================================

def print_run(snapshot: Dict, block: Dict, spread: Dict) -> None:
    print(f"\n{'=' * 72}\nDEZE RUN  ({snapshot['recorded_at']})\n{'=' * 72}")
    print(f"respondenten {snapshot['respondents']} | ideeën {snapshot['ideas_total']} "
          f"({snapshot['ideas_placed']} geplaatst, {snapshot['ideas_unplaced']} zonder attribuut)")
    print(f"boom: {snapshot['n_domains']} domeinen | {snapshot['n_facets']} facetten | "
          f"{snapshot['n_attributes']} attributen")

    thin, total = thin_facets(snapshot)
    threshold = max(T_KEEP_MIN, round(T_KEEP_SHARE * snapshot["respondents"]))
    print(f"facetten onder de codedrempel ({threshold} respondenten): "
          f"{thin} van {total} ({100 * thin / total:.0f}%)"
          f"{'   <-- boven de 25%-grens' if total and thin / total > 0.25 else ''}")

    print("\nfacetten naar bereik")
    for facet, reach in sorted(snapshot["facet_reach"].items(), key=lambda kv: -kv[1])[:12]:
        share = 100 * reach / snapshot["respondents"]
        n_attr = snapshot["facet_attributes"].get(facet, 0)
        print(f"  {reach:>5} ({share:>4.1f}%)  {n_attr:>3} attr  {facet}")
    if len(snapshot["facet_reach"]) > 12:
        print(f"  ... en {len(snapshot['facet_reach']) - 12} facetten meer")

    print(f"\nblokinvariant  {block['violations']} overtredingen op "
          f"{block['keys']} (domein, label)-sleutels"
          f"{'   <-- BUG: identiek label, ander attribuut' if block['violations'] else '   (zoals verwacht)'}")

    print(f"geërfde spreiding  {spread['pct']}%  ({spread['minority']} van "
          f"{spread['repeated_ideas']} herhaalde spans op de minderheidskant; "
          f"{spread['split_spans']} spans vallen uiteen, waarvan "
          f"{spread['across_domains']} over domeinen en {spread['within_domain']} "
          f"binnen één domein)")
    print("  dit is variantie die step 4 erft van step 3, geen ruisvloer van step 4 zelf")
    if spread["detail"]:
        print("\n  de uiteenvallende spans, naar frequentie")
        for span, n_dom, n_attr, n in spread["detail"][:8]:
            print(f"    {n:>3}x  {span[:40]:<40} {n_dom} domein(en), {n_attr} attributen")


def print_comparison(history: List[Dict]) -> None:
    print(f"\n{'=' * 72}\nOVER {len(history)} RUNS\n{'=' * 72}")

    print(f"{'run':<22}{'dom':>5}{'facet':>7}{'attr':>7}{'ideeën':>8}{'erf%':>8}{'dun%':>7}")
    for snap in history:
        thin, total = thin_facets(snap)
        print(f"{snap['recorded_at']:<22}{snap['n_domains']:>5}{snap['n_facets']:>7}"
              f"{snap['n_attributes']:>7}{snap['ideas_placed']:>8}"
              f"{inherited_spread(snap)['pct']:>8}"
              f"{(100 * thin / total if total else 0):>7.0f}")

    # Alle paren, niet alleen opeenvolgende: bij vijf runs zegt het gemiddelde
    # over tien paren meer dan vier losse getallen, en een enkele uitschieter
    # valt pas op als je hem tegen de rest kunt leggen.
    pairs = [(a, b) for i, a in enumerate(history) for b in history[i + 1:]]

    overlaps = [len(set(a["labels"]["attribuut"]) & set(b["labels"]["attribuut"]))
                for a, b in pairs]
    sizes = [len(s["labels"]["attribuut"]) for s in history]
    if overlaps and min(overlaps) < min(sizes):
        print(f"\n  LET OP: runs delen niet dezelfde ideeën "
              f"(overlap {min(overlaps)}-{max(overlaps)} tegen {min(sizes)}-{max(sizes)} "
              f"geplaatst). Is step 3 tussendoor opnieuw gedraaid? Dan meet dit "
              f"twee verschillende verzamelingen en zijn de ARI's beneden ongeldig.")

    print(f"\nindelingsstabiliteit over alle {len(pairs)} runparen (Adjusted Rand Index)")
    print("  1,00 = ideeën liggen identiek gegroepeerd; 0,00 = toeval")
    for level in LEVELS:
        values = [adjusted_rand_index(a["labels"][level], b["labels"][level])
                  for a, b in pairs]
        values = [v for v in values if v == v]  # NaN eruit
        if not values:
            continue
        mean = sum(values) / len(values)
        print(f"  {level:<11} gemiddeld {mean:.3f}   "
              f"laagste {min(values):.3f}   hoogste {max(values):.3f}")

    print("\n  per paar")
    for a, b in pairs:
        cells = "  ".join(
            f"{level[:4]} {adjusted_rand_index(a['labels'][level], b['labels'][level]):.3f}"
            for level in LEVELS)
        print(f"    {a['recorded_at'][-8:]} -> {b['recorded_at'][-8:]}   {cells}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE)
    cache_manager = CacheManager()

    taxonomy = cache_manager.load_metadata_from_cache(
        FILENAME, "taxonomy", variable_key, models.TaxonomyResultsCache)
    classified = cache_manager.load_from_cache(
        FILENAME, "taxonomy_classified", variable_key, models.TaxonomyClassifiedModel)

    if not taxonomy or not classified:
        print("Geen step-4-cache voor deze dataset+variabele. Draai step 4 eerst.")
        return

    snapshot = build_snapshot(classified, taxonomy)
    print_run(snapshot, block_move_violations(snapshot), inherited_spread(snapshot))

    # Eerst wegschrijven, daarna teruglezen: de snapshot van deze run moet de
    # volgende overleven, en die overschrijft de cache waaruit hij berekend is.
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    history = [
        s for s in (json.loads(line) for line in
                    SNAPSHOT_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
        if s.get("filename") == FILENAME and s.get("variable") == VARIABLE
    ]
    if len(history) > 1:
        print_comparison(history)
    else:
        print(f"\nEerste snapshot vastgelegd in {SNAPSHOT_FILE.name}. Draai step 4 "
              f"opnieuw en draai dit script daarna weer voor de vergelijking.")


if __name__ == "__main__":
    main()
