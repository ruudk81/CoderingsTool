#%%

"""
Test: Raw API call vs Instructor-wrapped call for quality filter.
Sends the same prompt to gpt-5-nano with and without instructor schema.
"""

import sys
import asyncio
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import nest_asyncio
nest_asyncio.apply()

from openai import AsyncOpenAI
from config import OPENAI_API_KEY, get_model, DEFAULT_LANGUAGE, get_reasoning_params
from utils.llm import create_client, llm_create_async

# Import the experimental prompt and schema
try:
    from development.step_2_qualityFilter.prompts_exp import GRADER_INSTRUCTIONS, QualityFilterLLMResponseExp
except ImportError:
    from prompts_exp import GRADER_INSTRUCTIONS, QualityFilterLLMResponseExp

# =============================================================================
# CONFIG
# =============================================================================
MODEL = get_model("nano")

# Load actual var_lab from test data
from utils import dataLoader
try:
    from development.test_data import TEST_DATA
except ImportError:
    from test_data import TEST_DATA
loader = dataLoader.DataLoader(data_dir=str(project_root / "data"), verbose=False)
VAR_LAB = loader.get_varlab(filename=TEST_DATA.filename, var_name=TEST_DATA.var_name)
print(f"Var lab: {VAR_LAB}")

# Test responses — mix of clearly meaningful + borderline
TEST_RESPONSES = [
    ("ID_1", "Ik vind de variatie in muziek altijd prima, we gaan met vrienden, en datgene wat ngng ons heeft geboden maakte het perfect."),
    ("ID_2", "Geweldige sfeer, leuke bands, fijne plek op camping, gemoedelijk, alle leeftijden door elkaar."),
    ("ID_3", "Geen specifieke reden."),
    ("ID_4", "People seemed very nice, the location was filled but not overcrowded."),
    ("ID_5", "asdf qwerty"),
    ("ID_6", "Weet ik niet"),
    ("ID_7", "De line-up was gevarieerd en heel leuk; vooral joost, olivia en tom."),
    ("ID_8", "Was leuk."),
    ("ID_9", "N/A"),
    ("ID_10", "Leuke sfeer, leuke optredens, jaarlijks extra vrienden uitje en het voelt als thuiskomen."),
]


def build_prompt(response_text: str) -> str:
    return GRADER_INSTRUCTIONS.format(
        language=DEFAULT_LANGUAGE,
        var_lab=VAR_LAB,
        response_text=response_text,
    )


# =============================================================================
# RAW API CALL (no instructor, no schema)
# =============================================================================
async def call_raw(response_id: str, response_text: str) -> str:
    """Call the API directly without instructor — get raw text back."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    prompt = build_prompt(response_text)

    result = await client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=100,
        reasoning={"effort": "minimal"},
    )
    # Extract text from response
    raw_text = result.output_text if hasattr(result, 'output_text') else str(result)
    return raw_text.strip()


# =============================================================================
# INSTRUCTOR CALL — default mode (schema injected into API)
# =============================================================================
async def call_instructor(response_id: str, response_text: str):
    """Call through instructor with the Pydantic response model."""
    client = create_client(model=MODEL, async_mode=True)
    prompt = build_prompt(response_text)

    response = await llm_create_async(
        client=client,
        model=MODEL,
        response_model=QualityFilterLLMResponseExp,
        prompt=prompt,
        temperature=0.0,
        max_tokens=100,
        track_usage=False,
        **get_reasoning_params(MODEL),
    )
    return response.quality_filter_code


# =============================================================================
# RAW + PARSE: raw API call, then parse into Pydantic manually
# =============================================================================
def parse_raw_response(raw_text: str):
    """Parse raw LLM text into the same type as instructor would return."""
    text = raw_text.strip().lower()
    if text == "null" or text == "none" or text == "":
        return None
    try:
        code = int(text)
        if code in (99999997, 99999999):
            return code
        return None  # unexpected code, treat as keep
    except ValueError:
        # Try to find a number in the text
        import re
        match = re.search(r'(99999997|99999999)', text)
        if match:
            return int(match.group(1))
        if "null" in text:
            return None
        return None  # can't parse, default to keep


# =============================================================================
# MAIN
# =============================================================================
async def main():
    print(f"Model: {MODEL}")
    print(f"Schema: {QualityFilterLLMResponseExp.model_json_schema()}")
    print(f"{'='*100}")
    print(f"{'ID':<8} {'Raw text':<14} {'Parsed':<14} {'Instructor':<14} {'Match?':<8} Response")
    print(f"{'='*100}")

    for resp_id, resp_text in TEST_RESPONSES:
        raw_result = await call_raw(resp_id, resp_text)
        parsed_result = parse_raw_response(raw_result)
        instructor_result = await call_instructor(resp_id, resp_text)

        match = "  ✓" if str(instructor_result) == str(parsed_result) else "  ✗"
        display_text = resp_text[:35] + "..." if len(resp_text) > 35 else resp_text

        print(f"{resp_id:<8} {str(raw_result):<14} {str(parsed_result):<14} {str(instructor_result):<14} {match:<8} {display_text}")

    print(f"{'='*100}")


if __name__ == "__main__":
    asyncio.run(main())
