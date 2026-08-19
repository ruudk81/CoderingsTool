"""De schrijfprompt van de v1-keten — alleen `run_codeGenerator.py` hieronder
gebruikt hem nog.

Het verschil met de productieprompt (`..prompts_writer`) is één regel: daar moet
de richting leesbaar zijn in naam én definitie, hier hoeft dat niet. Dat is geen
achterstand die nog weggewerkt moet worden maar een gemeten eigenschap van deze
keten — 19 richtingloze namen op 42 codes (2026-08-14). Trek je de twee gelijk,
dan is die meting niet meer te reproduceren.

De omliggende machinerie (`_code_block`, `_ordered`, `make_writer_model`,
`CodeText`, `WriterResult`) is niet versiegebonden en wordt gedeeld.
"""
from __future__ import annotations

from typing import List, Optional

from ..prompts_common import INSTRUCTOR_HINT
from ..prompts_writer import _code_block, _ordered


def build_writer_prompt_v1(
    shapes, concept_by_id, dimension_diagnostic: str, language: str,
    taken_names: Optional[List[str]] = None,
) -> str:
    inventory = "\n".join(_code_block(shape, concept_by_id) for shape in _ordered(shapes))

    taken_block = ""
    if taken_names:
        names = "\n".join(f"- {name}" for name in sorted(set(taken_names)))
        taken_block = f"""

These code names are already used elsewhere in this codebook and are OFF LIMITS —
do not reuse any of them for a code below, even if it would otherwise fit:
{names}"""

    return f"""You are writing the final codebook. The set of codes is already fixed: there are
{len(shapes)} of them, each with its direction already decided from the data. Write the
text for each one. Do NOT add, remove or merge codes.

For every code, write:
- a name (3-5 words) in {language}
- a definition: one interpretive sentence that reads like an analyst conclusion
- a diagnostic test completing this stem: "{dimension_diagnostic}"
- typical indicators: words or phrases that signal this code
- a boundary note against the nearest competing code in this list

Two rules:
1. A code name must not claim territory another code already owns — neither
   another code in the list below, nor one of the already-taken names listed
   further down, if any. If one code covers a specific topic and another
   covers the broader family it belongs to, name the broader one for what is
   left, not for the family.
2. If the source topics grouped under a code share nothing you can name honestly,
   set nameable to false. Do not invent an umbrella term to cover unrelated items.

Codes:
{inventory}
{taken_block}

{INSTRUCTOR_HINT}"""
