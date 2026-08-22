"""Wat het Overig-plafond telt, en wat het als defect moet melden.

De poort staat op `overig_idea_share_pct <= 10.0` en bewaakt hoeveel materiaal
het HOOFDcodeboek niet plaatste. Sinds 2026-08-22 kan dat materiaal op twee
plekken staan: ononderscheiden in Overig zelf, of in een benoemd kind eronder.
Deze tests leggen vast dat het plafond ze allebei telt, dat een kind daarbij
alleen zijn eigen valentiepool meebrengt, en dat een kapotte ouder-kindkoppeling
een gemeld defect is in plaats van een code die stil als hoofdcode meetelt.
"""
from models import ConsolidatedCode

from .codebook_verifier import build_scorecard, format_scorecard

OVERIG = "Overig"


def _taxonomie(attributen, toewijzingen, valenties=None):
    """Eén domein met `attributen` (id, naam), een idee→attribuutnaam-toewijzing
    en per idee een valentieteken. Genoeg voor de scorecard, die alles via
    `_attr` leest en dus net zo goed met dicts werkt als met de step-4-modellen.
    """
    return {"D1": {
        "attributes": {"domein": [{"attribute_id": i, "attribute_name": n}
                                  for i, n in attributen]},
        "attribute_assignments": toewijzingen,
        "attribute_valence": valenties or {},
    }}


def _code(naam, valence, bronnen, code_id="", parent=None):
    return ConsolidatedCode(
        code_name=naam, definition="d", diagnostic_test=f"test {naam}",
        valence=valence, typical_indicators=[], source_attributes=bronnen,
        code_id=code_id, parent_code_id=parent)


def _ideeen(prefix, attribuut, aantal, teken):
    """`aantal` ideeën op één attribuut met één valentieteken."""
    toewijzing = {f"{prefix}{k}": attribuut for k in range(aantal)}
    valentie = {f"{prefix}{k}": teken for k in range(aantal)}
    return toewijzing, valentie


def _set():
    """Eén groot onderwerp met een dunne tegenpool, plus een wees.

    `Groot onderwerp` draagt 200 positieve en 6 negatieve ideeën: de positieve
    pool is een hoofdcode, de negatieve haalde de drempel niet en hangt als kind
    onder Overig. `Wees` noemt geen enkele code, dus die veegt Overig zelf op.
    Dat is precies de vorm waarin een naïeve telling ontspoort — het kind deelt
    zijn bronattribuut met een hoofdcode van 200 ideeën.
    """
    t1, v1 = _ideeen("p", "Groot onderwerp", 200, "+")
    t2, v2 = _ideeen("n", "Groot onderwerp", 6, "-")
    t3, v3 = _ideeen("w", "Wees", 2, "")
    taxonomie = _taxonomie(
        [("A1", "Groot onderwerp"), ("A2", "Wees")],
        {**t1, **t2, **t3}, {**v1, **v2, **v3})
    codes = [
        _code("Groot onderwerp positief", "positive", ["Groot onderwerp"], "K1"),
        _code("Overige kritiek op groot onderwerp", "negative",
              ["Groot onderwerp"], "K2", parent="K3"),
        _code(OVERIG, "neutral", ["Wees"], "K3"),
    ]
    return codes, taxonomie


def test_de_poort_telt_de_kinderen_mee():
    """Ouder én kinderen staan buiten het hoofdcodeboek, en dat is wat het
    plafond bewaakt. Telde de poort alleen de kale ouder, dan zou materiaal
    onder Overig hangen zonder dat de poort het ziet."""
    codes, taxonomie = _set()
    sc = build_scorecard(codes, taxonomie, OVERIG)

    assert sc.overig_parent_idea_count == 2
    assert sc.overig_child_idea_count == 6
    assert sc.overig_child_code_names == ["Overige kritiek op groot onderwerp"]
    assert sc.overig_idea_share_pct == round(8 / 208 * 100, 1)


def test_een_kind_telt_alleen_zijn_eigen_valentiepool():
    """Een kind bezit één POOL van zijn bronattributen, niet het attribuut.

    Zijn bronattribuut is in de regel ook bron van een hoofdcode — op set 7
    gold dat voor 12 van de 17 kindattributen. Telt de poort het attribuut in
    plaats van de pool, dan claimt het kind alle ideeën van die hoofdcode: op
    set 7 gaf dat 55,5% in plaats van 3,6%, en hier 206 in plaats van 6.
    """
    codes, taxonomie = _set()
    sc = build_scorecard(codes, taxonomie, OVERIG)

    assert sc.overig_child_idea_count == 6, (
        "het kind sleept de 200 positieve ideeën van zijn bronattribuut mee")


def test_een_codeboek_met_kinderen_faalt_niet_ten_onrechte():
    """De poort mag niet vuren op de verbetering zelf. Dezelfde respondenten
    stonden vóór de kinderen ononderscheiden in Overig; nu dragen ze een naam
    en een richting. Het plafond hoort daar hetzelfde over te zeggen, niet
    strenger te worden."""
    codes, taxonomie = _set()
    sc = build_scorecard(codes, taxonomie, OVERIG)

    assert sc.overig_idea_share_pct <= 10.0
    assert sc.passed, sc.failure_reasons()


def test_de_scorecard_toont_beide_helften():
    """Welke helft de poort ook telt, een mens moet ze allebei kunnen aflezen.
    Eén samengesteld getal zonder zijn delen is wat dit besluit moeilijk
    maakte."""
    codes, taxonomie = _set()
    tekst = format_scorecard(build_scorecard(codes, taxonomie, OVERIG))

    assert "2 idea(s)" in tekst, tekst
    assert "6 idea(s)" in tekst, tekst
    assert "1 child code(s)" in tekst, tekst


def test_een_kind_zonder_ouder_is_een_defect():
    """`models.py` negeert onbekende init-kwargs stilzwijgend, dus een typefout
    als `parent_code=` levert een ouderloze code op zonder enige fout. Die code
    telt dan als gewone hoofdcode mee en valt buiten het Overig-plafond. De
    verifier is de plek waar dat alsnog opvalt."""
    codes, taxonomie = _set()
    kind = codes[1]
    kind.parent_code_id = None
    sc = build_scorecard(codes, taxonomie, OVERIG, child_code_ids={"K2"})

    assert sc.children_without_parent == ["Overige kritiek op groot onderwerp"]
    assert not sc.passed
    assert any("parent" in r for r in sc.failure_reasons()), sc.failure_reasons()


def test_een_ouderverwijzing_naar_een_onbekende_code_is_een_defect():
    """De spiegelfout: een `parent_code_id` die nergens op uitkomt. De code
    hangt dan aan niets, telt niet mee onder Overig, en niets zegt het."""
    codes, taxonomie = _set()
    codes[1].parent_code_id = "K99"
    sc = build_scorecard(codes, taxonomie, OVERIG)

    assert sc.dangling_parent_refs == ["Overige kritiek op groot onderwerp"]
    assert not sc.passed


def test_zonder_kinderen_telt_de_poort_wat_hij_altijd_telde():
    """Een codeboek zonder kinderen mag geen ander getal krijgen dan voorheen —
    anders zou de verandering een meting over de grens heen breken."""
    t1, v1 = _ideeen("p", "Groot onderwerp", 90, "+")
    t3, v3 = _ideeen("w", "Wees", 10, "")
    taxonomie = _taxonomie([("A1", "Groot onderwerp"), ("A2", "Wees")],
                           {**t1, **t3}, {**v1, **v3})
    codes = [_code("Groot onderwerp", "positive", ["Groot onderwerp"], "K1"),
             _code(OVERIG, "neutral", ["Wees"], "K2")]
    sc = build_scorecard(codes, taxonomie, OVERIG)

    assert sc.overig_idea_share_pct == 10.0
    assert sc.overig_child_idea_count == 0
    assert sc.passed


def _drie_polen():
    """Wat `POLES="three"` oplevert: één onderwerp met drie polen.

    `build_shapes(two_pole=False)` splitst een groep in positief/neutraal/
    negatief, en élke pool kan onder de drempel zakken en kind worden. Hier is
    de neutrale pool het kind, naast twee hoofdcodes op hetzelfde attribuut.
    """
    t1, v1 = _ideeen("p", "Groot onderwerp", 200, "+")
    t2, v2 = _ideeen("n", "Groot onderwerp", 6, "-")
    t3, v3 = _ideeen("u", "Groot onderwerp", 5, "")
    t4, v4 = _ideeen("w", "Wees", 2, "")
    taxonomie = _taxonomie(
        [("A1", "Groot onderwerp"), ("A2", "Wees")],
        {**t1, **t2, **t3, **t4}, {**v1, **v2, **v3, **v4})
    codes = [
        _code("Lof", "positive", ["Groot onderwerp"], "K1"),
        _code("Kritiek", "negative", ["Groot onderwerp"], "K2"),
        _code("Overige opmerkingen", "neutral", ["Groot onderwerp"], "K3", parent="K4"),
        _code(OVERIG, "neutral", ["Wees"], "K4"),
    ]
    return codes, taxonomie


def test_een_neutraal_kind_telt_alleen_zijn_neutrale_pool():
    """De poort mag een neutraal KIND niet zijn hele attribuut laten claimen.

    `POLES="three"` staat in de runner en levert neutrale polen op die kind
    kunnen worden. Claimt zo'n kind het attribuut, dan sleept het de ideeën van
    de positieve en negatieve hoofdcode ernaast mee — precies de fout die deze
    taak bij de attribuuttelling verwierp. Op de set-7-runset met `POLES="three"`
    scheelt dat 22,6% (FAIL) tegen 6,2% (PASS).
    """
    codes, taxonomie = _drie_polen()
    sc = build_scorecard(codes, taxonomie, OVERIG)

    assert sc.overig_child_idea_count == 5, (
        "het neutrale kind claimt de ideeën van de codes naast zich")
    assert sc.overig_idea_share_pct == round(7 / 213 * 100, 1)
    assert sc.passed, sc.failure_reasons()


def test_een_non_negative_code_verwacht_zijn_negatieve_ideeen_niet():
    """De mini-codetelling is een levend mechanisme en verandert hier van gedrag.

    Een `non_negative` code dekt positief plus neutraal. Telde zijn verwachting
    ook de negatieve ideeën van zijn bronattribuut mee, dan kon zo'n code nooit
    onder de bodem uitkomen — de waarschuwing die over-differentiatie moet
    aanwijzen zweeg dan juist bij de codes die hem nodig hadden.
    """
    t1, v1 = _ideeen("p", "Onderwerp", 1, "+")
    t2, v2 = _ideeen("u", "Onderwerp", 1, "")
    t3, v3 = _ideeen("n", "Onderwerp", 50, "-")
    t4, v4 = _ideeen("w", "Wees", 2, "")
    taxonomie = _taxonomie([("A1", "Onderwerp"), ("A2", "Wees")],
                           {**t1, **t2, **t3, **t4}, {**v1, **v2, **v3, **v4})
    codes = [_code("Positieve kant", "non_negative", ["Onderwerp"], "K1"),
             _code("Negatieve kant", "negative", ["Onderwerp"], "K2"),
             _code(OVERIG, "neutral", ["Wees"], "K3")]
    sc = build_scorecard(codes, taxonomie, OVERIG)

    mini = {m.code_name: m.expected_ideas for m in sc.mini_codes}
    assert mini.get("Positieve kant") == 2, sc.mini_codes


def test_een_neutrale_hoofdcode_verwacht_wel_zijn_hele_attribuut():
    """De andere helft van dezelfde splitsing, en de reden dat er twee regels
    zijn. Een neutrale HOOFDcode is per `models.py` dimensioneel — geen pool
    haalde de poort — en dekt zijn attribuut ongeacht richting. Zou hij hier
    strikt op zijn neutrale pool geteld worden, dan meldde de scorecard hem als
    mini-code terwijl hij 52 ideeën draagt."""
    t1, v1 = _ideeen("p", "Onderwerp", 1, "+")
    t2, v2 = _ideeen("u", "Onderwerp", 1, "")
    t3, v3 = _ideeen("n", "Onderwerp", 50, "-")
    t4, v4 = _ideeen("w", "Wees", 2, "")
    taxonomie = _taxonomie([("A1", "Onderwerp"), ("A2", "Wees")],
                           {**t1, **t2, **t3, **t4}, {**v1, **v2, **v3, **v4})
    codes = [_code("Dimensionele code", "neutral", ["Onderwerp"], "K1"),
             _code(OVERIG, "neutral", ["Wees"], "K2")]
    sc = build_scorecard(codes, taxonomie, OVERIG)

    assert [m.code_name for m in sc.mini_codes] == []
