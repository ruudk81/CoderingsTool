"""
Experimental Prompts for Step 1: Preprocess (Spell Checking)

This file contains the prompts used by spellChecker.py.
Modify these prompts to experiment with different spell checking approaches.

Original source: src/prompts.py (STEP 1: SPELL CHECKING section)

Response models (Pydantic) are co-located with their prompts following the
migrate-output-schema pattern - instructor uses Field(description=...) to
communicate schema to the LLM.
"""

from typing import List
from pydantic import BaseModel, Field

# =============================================================================
# RESPONSE MODELS (co-located with prompts for instructor)
# =============================================================================

class CorrectionItem(BaseModel):
    """A single spell correction result."""
    respondent_id: str = Field(
        description="The respondent ID from the correction task"
    )
    corrected_response: str = Field(
        description="The fully corrected response with all spelling mistakes fixed"
    )


class LLMCorrectionResponse(BaseModel):
    """Structured output for spell check corrections."""
    corrections: List[CorrectionItem] = Field(
        description="List of corrections, one for each task in the input"
    )


class CompoundVerdict(BaseModel):
    """One decision about one candidate pair."""
    pair: str = Field(
        description="The candidate pair, copied exactly as it was given"
    )
    join: bool = Field(
        description="True if these two words must be written as a single word here"
    )


class SplitCompoundResponse(BaseModel):
    """Structured output for the split-compound decision. Verdicts only: the edit
    itself is made in code, so this phase can never reword the response."""
    verdicts: List[CompoundVerdict] = Field(
        description="One verdict per candidate pair, in the order they were given"
    )


# =============================================================================
# STEP 1: SPELL CHECKING
# =============================================================================

SPELLCHECK_INSTRUCTIONS = """
You are a {language} language expert specializing in correcting misspelled words in open-ended survey responses.
Your task is to process correction tasks for responses that contain placeholder tokens indicating spelling mistakes.

First, here is the survey question that the responses are answering:
<survey_question>
{var_lab}
</survey_question>

For each correction task, you will receive:
- A sentence with one or more <oov_word> placeholders
- A list of misspelled words, in the same order as the placeholders
- A list of suggested corrections, in the same order

Follow these rules when making corrections:
1. Replace each <oov_word> placeholder with the best possible correction of the corresponding misspelled word.
2. Consider the meaning and context of the survey question when choosing corrections.
3. If a better correction exists than the ones provided, use that instead.
4. You may split a misspelled word into two words only if the split preserves the intended meaning and fits grammatically.
5. If no suitable correction is possible, use "[NO RESPONSE]" as the corrected sentence for that task.

Here are the correction tasks to process:
<correction_tasks>
{tasks}
</correction_tasks>

Additional guidelines:
- Pay close attention to the context and meaning of each response when making corrections.
- Ensure that your corrections maintain the original intent of the respondent.
- If a suggested correction doesn't fit the context, consider alternative corrections that preserve the meaning.

Begin processing the correction tasks now and provide your output as valid JSON following the response schema provided.
"""


# =============================================================================
# STEP 1: SPLIT COMPOUNDS
# =============================================================================

SPLIT_COMPOUND_INSTRUCTIONS = """
You are a {language} language expert judging open-ended survey responses.

Each candidate below is a pair of adjacent words in the response. For each pair, decide
whether those two words are one compound word that the respondent wrote apart by mistake,
or two separate words that are correctly written apart.

First, here is the survey question that the response is answering:
<survey_question>
{var_lab}
</survey_question>

Here is the response:
<response>
{response}
</response>

Here are the candidate pairs:
<candidates>
{candidates}
</candidates>

Judge each pair on its own, using these rules:
1. Answer true only when the two words together name one single thing or one single
   property, and writing them apart is a spelling mistake in this sentence.
2. Answer false when the first word describes the second as a separate word, when the
   two words belong to different parts of the sentence, or when both spellings are
   acceptable.
3. Judge the sentence as the respondent meant it. Read the pair in its context in the
   response, not in isolation.
4. Copy each pair back exactly as it was given, and return one verdict per candidate.

Provide your output as valid JSON following the response schema provided.
"""
