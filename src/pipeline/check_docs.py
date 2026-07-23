"""Check the dev/ documentation against DEV_DOC_STANDARD.md. Deterministic, no LLM.

A manual "last verified" line does not work — it goes stale silently, which is how
three PROCESSING.md files ended up claiming 2026-04-05 while their code moved on for
two months. Git already knows both dates, so this compares them instead.

Known limit of the staleness check: it compares commit dates, so ANY commit touching
a doc clears the flag — including a typo fix. It tells you the code moved after the
doc was last touched, which is a reason to look, not proof that the doc is wrong.
Editing a doc without actually checking it against the code defeats it.

Run from the repo root or from src/:
    python src/pipeline/check_docs.py

Exit code 0 when clean, 1 when anything is reported, so it can gate a commit.
"""
import hashlib
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PIPELINE = Path(__file__).parent
CORE = {"CLAUDE.md", "ARCHITECTURE.md", "CACHE_LOGIC.md", "PROCESSING.md"}
ALLOWED = CORE | {"WORK.md"}
DESIGN = re.compile(r"^DESIGN_[A-Z0-9_]+\.md$")
# Only top-level sections. Numbered `###` are legitimate: PROCESSING.md's Contract
# section is defined as a numbered list of principles.
NUMBERED_HEADING = re.compile(r"^## \d+\.\s", re.M)
MOJIBAKE = re.compile("[�]")


def strip_code_blocks(text: str) -> str:
    """Drop fenced blocks. A `# comment` inside one is not a markdown heading."""
    return re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)


def git_date(path: Path) -> str:
    """Last commit date for a path, '' if untracked."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
        capture_output=True, text=True, cwd=PIPELINE,
    )
    return out.stdout.strip()


def newest_code_date(step_dir: Path) -> str:
    dates = [git_date(p) for p in step_dir.glob("*.py")]
    return max((d for d in dates if d), default="")


def check() -> list[str]:
    findings: list[str] = []
    by_hash: dict[str, list[Path]] = defaultdict(list)

    for step_dir in sorted(PIPELINE.glob("step_*")):
        dev = step_dir / "dev"
        if not dev.is_dir():
            findings.append(f"{step_dir.name}: geen dev/ map")
            continue

        docs = sorted(dev.glob("*.md"))
        names = {d.name for d in docs}
        code_date = newest_code_date(step_dir)

        if "CLAUDE.md" not in names:
            findings.append(f"{step_dir.name}: CLAUDE.md ontbreekt (altijd verplicht)")

        for doc in docs:
            rel = f"{step_dir.name}/dev/{doc.name}"

            if doc.name not in ALLOWED and not DESIGN.match(doc.name):
                findings.append(f"{rel}: staat niet in de toegestane lijst "
                                f"(kern, WORK.md of DESIGN_<ONDERWERP>.md)")

            doc_date = git_date(doc)
            if doc_date and code_date and doc_date < code_date:
                findings.append(f"{rel}: doc {doc_date} is ouder dan de code "
                                f"in deze stap ({code_date})")

            text = doc.read_text(encoding="utf-8")
            by_hash[hashlib.md5(text.encode()).hexdigest()].append(doc)
            prose = strip_code_blocks(text)

            first = text.split("\n", 1)[0]
            if doc.name in CORE or doc.name == "WORK.md":
                if not re.match(r"^# Step \d+ — ", first):
                    findings.append(f"{rel}: titel wijkt af van '# Step N — Naam'")
            elif not first.endswith("— Design"):
                findings.append(f"{rel}: titel van een DESIGN-doc moet eindigen op '— Design'")

            if NUMBERED_HEADING.search(prose):
                findings.append(f"{rel}: genummerde secties (## 1. …) — gebruik gewone koppen")

            if MOJIBAKE.search(prose):
                findings.append(f"{rel}: kapotte tekens (mojibake) — bestand niet als UTF-8 opgeslagen")

            if prose.count("\n# ") >= 1:
                findings.append(f"{rel}: meer dan één titel op niveau 1 — "
                                f"splits dit in aparte documenten")

            for link in re.findall(r"\]\(([^)#][^)]*\.md)\)", prose):
                if not (doc.parent / link).exists():
                    findings.append(f"{rel}: kapotte link -> {link}")

    for paths in by_hash.values():
        if len(paths) > 1:
            rels = ", ".join(str(p.relative_to(PIPELINE)) for p in paths)
            findings.append(f"byte-identieke documenten: {rels}")

    return findings


def main() -> int:
    findings = check()
    if not findings:
        print("dev-docs: geen meldingen")
        return 0
    print(f"dev-docs: {len(findings)} melding(en)\n")
    for f in findings:
        print(f"  {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
