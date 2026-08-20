"""Wat elke promptmodule in step 5 nodig heeft.

Deze twee stonden in `prompts_relations.py` omdat dat toevallig de eerste
promptmodule was. Elke andere promptmodule importeerde ze daaruit, waardoor de
relatiefase — die sinds de v2-promotie niet meer draait — niet weg kon zonder
de rest mee te nemen.
"""
from __future__ import annotations

import hashlib

# Instructor houdt zich niet aan `Field(description=...)` alleen: zonder deze
# zin aan het eind van de prompt faalt een groot deel van de taken op het
# responsemodel. Letterlijk overnemen, nooit hertypen.
INSTRUCTOR_HINT = (
    "provide your output as valid JSON following the response schema provided"
)


def _shuffled(concepts, salt: str = ""):
    """Concepts in a deterministic order unrelated to prevalence OR domain.

    Both the prompt text and the response model's enum must use this instead of
    the caller's order: `concepts` arrives prevalence-sorted (build_inventory's
    `(-n_resp, name)`), and that order — whether in prose or in a JSON schema
    enum — is itself a signal about how often something occurs. Sorting by
    attribute_id instead fixes that, but opens a second, subtler channel:
    identity.py mints A# sequentially PER DOMAIN, so id order still groups
    domains into contiguous blocks — visible structure of exactly the kind this
    step exists to stop handing the model. Sorting by a hash of the id keeps the
    order reproducible across runs (the hash is a pure function of a stable id)
    while carrying neither signal.

    `salt` verschuift die volgorde reproduceerbaar. Dat is er voor het
    consensus-experiment, dat dezelfde vraag meermaals stelt en waarvoor de
    aanbiedingsvolgorde de enige beschikbare variatiebron is: redeneermodellen
    krijgen geen temperature-parameter. Zonder salt is het resultaat identiek
    aan de ongesalte versie.
    """
    return sorted(concepts,
                  key=lambda c: hashlib.md5((salt + c.attribute_id).encode()).hexdigest())
