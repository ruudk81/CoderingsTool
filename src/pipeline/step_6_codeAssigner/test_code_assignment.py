"""Tests voor de kandidaatkant van step 6: herkomstkaarten en kandidatenmenu.

Sinds 2026-08-22 kan het codeboek KINDEREN bevatten: volwaardige codes met een
eigen naam, definitie, indicatoren en valentie die via `parent_code_id` onder de
Overig-code hangen. Ze zijn volwaardige codes en doen dus gewoon mee in het
kandidatenmenu — maar ze mogen nooit de gegarandeerde thuiscode van een idee
zijn, en ze mogen nooit voor de Overig-ouder worden aangezien.

De productie-step-5 levert vandaag nog codeboeken ZONDER kinderen. Alles hier
moet daarom op zo'n codeboek exact hetzelfde doen als vóór deze wijziging; dat
is wat `test_kinderloos_codeboek_...` vastlegt.
"""

import pytest

from models import (
    ConsolidatedCode,
    DomainDescription,
    DomainResultModel,
    DomainSet,
)
from pipeline.step_6_codeAssigner.code_assignment import CodeAssigner
from pipeline.step_6_codeAssigner.config_codeAssigner import AssignmentConfig
from pipeline.step_6_codeAssigner.embedding_matcher import EmbeddingMatcher


# =============================================================================
# FIXTURES
# =============================================================================

def code(
    name: str,
    code_id: str,
    attr_ids,
    *,
    valence: str = "neutral",
    parent: str = None,
    definition: str = "d",
    indicators=None,
) -> ConsolidatedCode:
    return ConsolidatedCode(
        code_name=name, definition=definition, diagnostic_test="t",
        valence=valence, typical_indicators=indicators or [],
        code_id=code_id, parent_code_id=parent,
        source_attributes=[f"attr {a}" for a in attr_ids],
        source_attribute_ids=list(attr_ids),
    )


def mece_results(*attr_specs) -> dict:
    """(domein, facet, attribuutnaam, A#) → de structuur waaruit step 6 de
    naam→A#-resolver bouwt."""
    per_domein = {}
    for domain, facet, name, attr_id in attr_specs:
        res = per_domein.setdefault(domain, {"facets": {}})
        res["facets"].setdefault(facet, []).append(
            {"attribute_name": name, "attribute_id": attr_id})
    return {
        domain: DomainResultModel(
            partition_name=domain, n_labels=1, n_batches=1,
            attributes=data["facets"],
        )
        for domain, data in per_domein.items()
    }


def assigner(codes, results=None, attribute_assignments=None) -> CodeAssigner:
    """Een CodeAssigner met alleen de zuivere kant gevuld — geen LLM, geen
    embeddings, geen event loop."""
    return CodeAssigner(
        config=AssignmentConfig(),
        ideas_models=[],
        mece_results=results if results is not None else {},
        partition_set=DomainSet(partitions=[DomainDescription(
            partition_name="d", inclusion_definition="i",
            exclusion_definition="e", boundary_test="b",
            diagnostic_signals=[], concept_examples=[])]),
        codes=codes,
        attribute_assignments=attribute_assignments or {},
    )


# =============================================================================
# 1. DE THUISCODE IS NOOIT EEN KIND
# =============================================================================

def test_thuiscode_is_nooit_een_kind():
    """`_attr_to_code_idx` neemt de EERSTE code die een attribuut claimt. Staat
    een kind vooraan in de lijst, dan zou het gegarandeerde vangnet van een idee
    een minicode onder Overig worden — precies de code die niet bedoeld is als
    dekking, maar als restplaats."""
    codes = [
        code("Negatieve overige klantservice", "K31", ["A1"], parent="K9"),
        code("Service en advies", "K2", ["A1"]),
        code("Overig", "K9", []),
    ]
    a = assigner(codes)
    a._build_provenance_maps()
    assert a._attr_to_code_idx["A1"] == 1


def test_attribuut_dat_alleen_een_kind_claimt_heeft_geen_thuiscode():
    """En dan is er géén seeding. Dat is de bedoeling: het kandidatenmenu is
    niet leeg — het kind doet gewoon mee via de embeddingvoorfilter, en de
    no-fit-optie komt uit op de Overig-ouder van datzelfde kind. Het idee valt
    dus in het slechtste geval in dezelfde bak waar zijn kind onder hangt."""
    codes = [
        code("Negatieve overige klantservice", "K31", ["A1"], parent="K9"),
        code("Overig", "K9", []),
    ]
    a = assigner(codes, mece_results(("domein", "facet", "attr A1", "A1")))
    a._build_provenance_maps()
    assert "A1" not in a._attr_to_code_idx
    assert a._home_code_idx("domein", "attr A1") is None


def test_no_fit_komt_uit_op_de_ouder_niet_op_een_kind():
    """De catch-all wordt op NAAM herkend, en een kind draagt een door een LLM
    geschreven naam. Een kind dat toevallig een catch-all-woord treft mag de
    no-fit-bestemming niet kapen: de hiërarchie zit in het veld, dus een code
    met een `parent_code_id` kan de ouder niet zijn."""
    codes = [
        code("Other", "K31", ["A1"], parent="K9"),
        code("Overig", "K9", []),
    ]
    a = assigner(codes)
    a._build_provenance_maps()
    assert a._no_fit_resolves_to == ("K9", "Overig")


# =============================================================================
# 2. KINDEREN DOEN GEWOON MEE IN HET KANDIDATENMENU
# =============================================================================

def test_kinderen_staan_in_het_globale_id_menu():
    codes = [
        code("Service en advies", "K2", ["A1"]),
        code("Negatieve overige klantservice", "K31", ["A1"], parent="K9"),
        code("Overig", "K9", []),
    ]
    a = assigner(codes)
    a._build_provenance_maps()
    a._build_id_maps()
    assert a._id_to_code["C2"] == ("K31", "Negatieve overige klantservice")
    assert a._id_to_code[a._no_fit_id] == ("K9", "Overig")


def test_build_code_text_werkt_op_een_kind():
    """De voorfilter embedt elke code op naam | definitie | indicatoren. De
    kindschrijver levert alle drie, dus een kind heeft dezelfde embeddingtekst
    als elke andere code — geen leeg veld dat de cosinus laat ontsporen."""
    kind = code(
        "Negatieve overige klantservice", "K31", ["A1"], parent="K9",
        valence="negative", definition="Klachten over service die geen eigen kop kregen",
        indicators=["traag", "geen reactie"],
    )
    assert EmbeddingMatcher.build_code_text(kind) == (
        "Negatieve overige klantservice | "
        "Klachten over service die geen eigen kop kregen | traag, geen reactie"
    )


# =============================================================================
# DE PRODUCTIEVOORWAARDE
# =============================================================================

def test_kinderloos_codeboek_gedraagt_zich_exact_als_voorheen():
    """Productie draait step 5 nog zónder kinderen: `parent_code_id` is daar op
    élke code None. De voorwaarde van de gebruiker is dat step 6 op zo'n
    codeboek geen haar anders werkt dan vóór 2026-08-22.

    Deze test vergelijkt niet met een verwachting maar met het OUDE algoritme,
    hier in drie regels nagebouwd: de eerste code die een attribuut claimt, en
    de eerste code die een catch-all-naam draagt. Lekt kindbewustzijn ooit naar
    het kinderloze geval, dan lopen de twee uiteen en faalt deze test.
    """
    codes = [
        code("Milieu positief", "K1", ["A1", "A2"], valence="positive"),
        code("Milieu negatief", "K2", ["A1", "A2"], valence="negative"),
        code("Kosten", "K3", ["A3"]),
        code("Overig", "K4", ["A9"]),
    ]
    assert all(c.parent_code_id is None for c in codes)

    verwacht_attr = {}
    for i, c in enumerate(codes):
        for attr_id in c.source_attribute_ids:
            verwacht_attr.setdefault(attr_id, i)
    verwacht_overig = next(
        i for i, c in enumerate(codes) if c.code_name.strip().lower() == "overig")

    a = assigner(codes)
    a._build_provenance_maps()

    assert a._attr_to_code_idx == verwacht_attr
    assert a._overig_code_idx == verwacht_overig
    assert a._no_fit_resolves_to == (codes[verwacht_overig].code_id,
                                     codes[verwacht_overig].code_name)


@pytest.mark.parametrize("domein,naam,verwacht", [
    ("domein", "attr A1", 0),      # (domein, naam) → A1 → K1
    ("DOMEIN", "attr A1", 0),      # domeinsleutel is genormaliseerd
    ("ander", "attr A3", 2),       # naam structuurbreed uniek
    ("domein", "onbekend", None),
])
def test_kinderloos_codeboek_seeding_ongewijzigd(domein, naam, verwacht):
    """De tweede helft van dezelfde voorwaarde: de kandidaatzaaiing wijst op een
    kinderloos codeboek nog exact dezelfde thuiscode aan."""
    codes = [
        code("Milieu positief", "K1", ["A1", "A2"], valence="positive"),
        code("Milieu negatief", "K2", ["A1", "A2"], valence="negative"),
        code("Kosten", "K3", ["A3"]),
        code("Overig", "K4", []),
    ]
    results = mece_results(
        ("domein", "facet", "attr A1", "A1"),
        ("domein", "facet", "attr A2", "A2"),
        ("ander", "facet", "attr A3", "A3"),
    )
    a = assigner(codes, results)
    a._build_provenance_maps()
    assert a._home_code_idx(domein, naam) is verwacht


# =============================================================================
# 3. HET NULSIGNAAL MOET LEESBAAR ZIJN
# =============================================================================

def _output_met(codes, toewijzingen, capsys):
    from models import CodeAssignedModel, CodeAssignedSubmodel
    resp = CodeAssignedModel(respondent_id=1, response="r", response_ideas=[
        CodeAssignedSubmodel(idea_id=f"i{n}", idea="i", assigned_code=naam,
                             confidence=0.9)
        for n, naam in enumerate(toewijzingen)])
    a = assigner(codes)
    a._build_provenance_maps()
    a._print_assignment_summary([resp])
    return capsys.readouterr().out


def test_een_kind_zonder_toewijzingen_staat_er_met_zijn_nul_bij(capsys):
    """Het falsificatiesignaal uit de spec is dat een kind NUL ideeën vangt.
    Een nul die alleen bestaat als afwezigheid in een lijst is niet af te lezen,
    dus elk kind wordt genoemd — ook het lege."""
    codes = [
        code("Service en advies", "K2", ["A1"]),
        code("Vol kind", "K31", ["A1"], parent="K9", valence="negative"),
        code("Leeg kind", "K32", ["A2"], parent="K9", valence="negative"),
        code("Overig", "K9", []),
    ]
    uit = _output_met(codes, ["Service en advies", "Vol kind"], capsys)
    assert "CHILDREN OF 'Overig' (2)" in uit
    assert "Vol kind (negative): 1 ideas" in uit
    assert "Leeg kind (negative): 0 ideas   ← ZERO" in uit


def test_kinderloos_codeboek_drukt_geen_kindblok_af(capsys):
    """De productievoorwaarde, ook in de rapportage: op een codeboek zonder
    kinderen verandert er geen regel aan de uitvoer."""
    codes = [code("Service en advies", "K2", ["A1"]), code("Overig", "K9", [])]
    uit = _output_met(codes, ["Service en advies"], capsys)
    assert "CHILDREN OF" not in uit
