"""
promptViewer.py - Read back the prompts a step actually sent to the LLM.

Every step captures its prompts during a run (utils/promptPrinter.py) and writes
them to exports/prompts/ under the canonical name from utils/exportNaming.py.
This module renders one such file: the prompt text plus the tool definition
instructor injects alongside it, which together are the complete request.

A step's own view_prompts.py supplies the one thing that is step-specific — which
prompt_type maps to which response model — and calls render():

    from utils.promptViewer import render
    render(step=4, models=PROMPT_MODELS, test_data=TEST_DATA)

A mapping value is either a model class, or a callable taking the prompt entry's
metadata and returning a class. The callable form exists for step 3, where the
extraction model is built per dimension at runtime.

By default one instance per prompt_type is shown: a taxonomy run captures
hundreds of near-identical assignment prompts, and printing them all buries the
distinct ones. The header states how many instances each type has, so a single
example is never mistaken for the whole set.
"""

import json
from pathlib import Path
from typing import Callable, Optional, Union

from instructor.function_calls import openai_schema

from utils.exportNaming import export_filename, parse_export_filename

project_root = Path(__file__).parent.parent.parent
PROMPTS_DIR = project_root / "exports" / "prompts"
DATA_DIR = project_root / "data"

RULE = "=" * 100
THIN = "-" * 100

# Either a response model, or metadata -> response model for runtime-built ones.
ModelSource = Union[type, Callable[[dict], Optional[type]]]

# Printed first and in this order; everything else follows alphabetically. Keeps
# the model and its settings at eye level without hardcoding every step's keys.
LEAD_METADATA = ("model", "temperature", "max_tokens", "language")


def _resolve_model(models: dict, entry: dict) -> tuple:
    """Returns (model, note). Model is None when the type is unmapped."""
    prompt_type = entry.get("prompt_type", "unknown")
    source = models.get(prompt_type)

    if source is None:
        return None, f"no response model mapped for {prompt_type!r}"
    if callable(source) and not isinstance(source, type):
        model = source(entry.get("metadata", {}))
        if model is None:
            return None, f"builder for {prompt_type!r} returned no model"
        return model, f"{model.__name__} (built at runtime)"
    return source, source.__name__


def _schema_text(model) -> Optional[str]:
    """The tool definition instructor sends, or None if it cannot be built."""
    try:
        return json.dumps(openai_schema(model).openai_schema, indent=2)
    except Exception as exc:
        return f"(could not build schema: {exc})"


def _print_entry(entry: dict, index: int, total: int, models: dict,
                 instances: int) -> None:
    prompt_type = entry.get("prompt_type", "unknown")
    print(f"\n{RULE}")
    suffix = f" — 1 of {instances} instances" if instances > 1 else ""
    print(f"PROMPT {index}/{total}: {prompt_type}{suffix}")
    print(RULE)

    metadata = entry.get("metadata", {})
    print(f"Captured by: {entry.get('utility_name', '?')} "
          f"({entry.get('step_name', '?')})")
    for key in LEAD_METADATA:
        if key in metadata:
            print(f"{key.replace('_', ' ').capitalize():<12} {metadata[key]}")
    rest = {k: v for k, v in metadata.items() if k not in LEAD_METADATA}
    for key in sorted(rest):
        print(f"{key.replace('_', ' ').capitalize():<12} {rest[key]}")

    content = entry.get("prompt_content", "(no content)")
    print("\n[Prompt as sent]")
    print(THIN)
    print(content)
    print(THIN)

    model, note = _resolve_model(models, entry)
    print(f"\n[Response model] {note}")
    schema = _schema_text(model) if model is not None else None
    if schema:
        print("\n[Tool definition instructor injects]")
        print(THIN)
        print(schema)
        print(THIN)

    print("\n[Size]")
    print(f"  Prompt: {len(content):,} chars (~{len(content) // 4:,} tokens)")
    if schema:
        print(f"  Schema: {len(schema):,} chars (~{len(schema) // 4:,} tokens)")


def render_file(path: Path, models: dict, show_all: bool = False) -> None:
    """Render one saved prompts file. Assumes it exists."""
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])

    print(RULE)
    print(f"PROMPTS SENT — {path.name}")
    print(RULE)
    print(f"Session:      {data.get('session_id', 'unknown')}")
    print(f"Captured:     {data.get('capture_time', 'unknown')}")
    print(f"Total:        {len(prompts)} prompt(s)")

    if not prompts:
        print("\nThe file holds no prompts.")
        return

    counts = {}
    for entry in prompts:
        ptype = entry.get("prompt_type", "unknown")
        counts[ptype] = counts.get(ptype, 0) + 1

    print(f"\nBy type ({len(counts)} distinct):")
    for ptype in sorted(counts):
        mapped = "" if ptype in models else "   [no model mapped]"
        print(f"  {counts[ptype]:>5}x  {ptype}{mapped}")

    if show_all:
        selected = prompts
        print(f"\nShowing all {len(prompts)}.")
    else:
        first = {}
        for entry in prompts:
            first.setdefault(entry.get("prompt_type", "unknown"), entry)
        selected = list(first.values())
        print(f"\nShowing one example per type ({len(selected)} of "
              f"{len(prompts)}). Pass show_all=True for every instance.")

    for i, entry in enumerate(selected, 1):
        _print_entry(entry, i, len(selected), models,
                     counts[entry.get("prompt_type", "unknown")])


def _print_alternatives(step: int | str, wanted: Path) -> None:
    """After a miss, show which prompt files do exist — and for which run."""
    print(f"\nNo prompts file at: {wanted}")

    stems = [p.name for p in DATA_DIR.glob("*.sav")]
    found = []
    for path in sorted(PROMPTS_DIR.glob("*.json")):
        parsed = parse_export_filename(path.name, stems)
        if parsed and parsed.doctype == f"prompts_step{step}":
            found.append(parsed)

    if found:
        print(f"\nStep {step} prompts that were captured, for other runs:")
        for p in found:
            print(f"  {p.dataset}  {p.var_name}  n={p.sample}")
        print("\nPoint test_data.py at one of these, or run the step for the "
              "current one.")
    else:
        print(f"\nNo step-{step} prompts have been captured yet under the "
              f"canonical name. Files written before exportNaming.py landed "
              f"use the old convention and are not read back.")


def render(step: int | str, models: dict, test_data, show_all: bool = False) -> None:
    """Render the prompts the given step sent for the run in test_data."""
    name = export_filename(
        test_data.filename,
        test_data.var_name,
        test_data.sample_size,
        f"prompts_step{step}",
        "json",
    )
    path = PROMPTS_DIR / name

    print(RULE)
    print(f"STEP {step} — PROMPT INSPECTOR")
    print(RULE)
    print(f"Dataset:      {test_data.filename}")
    print(f"Variable:     {test_data.var_name}")
    print(f"Sample size:  {test_data.sample_size if test_data.sample_size is not None else 'full'}")

    if not path.exists():
        _print_alternatives(step, path)
        return

    print()
    render_file(path, models, show_all=show_all)
