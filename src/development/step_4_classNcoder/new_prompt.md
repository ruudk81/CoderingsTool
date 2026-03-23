P9 — Codebook Consolidation workflow (templatized)

## STEP 1 — PRE-STRUCTURE BY VALENCE (HARD GATE)

Valence Sensitivity Rule
- Generate separate codes for positive and negative phenomena.
- Do NOT combine praise and criticism into a single code.
- If the attributes contain both positive and negative aspects of similar phenomena, create distinct codes for each valence direction.


---

## STEP 2 — CLUSTER BY LATENT QUESTION (NOT TOPIC)

Instead of grouping by topic, group by:

**{domain_diagnostic}**

If two codes answer the same question → same cluster

---

## STEP 3 — AGGRESSIVE MERGING WITHIN CLUSTERS

Within each valence + question cluster:

Merge until:

> A coder would NEVER hesitate between remaining codes

### Strict Merge Rule:

If both can apply to the same sentence → **merge**

---

## STEP 4 — MECHANISM PURITY CHECK

For each code, ask:

Is this describing:

* a **value** (e.g., fair, responsible)
* a **functional property** (e.g., fast, easy to use)
* a **perception/judgment** (e.g., reliable, outdated)
* a **cause/reason** (e.g., due to specific actions or policies)

If mixed → SPLIT

---

## STEP 5 — NEIGHBOUR STRESS TEST

For every pair of same-valence codes:

Ask:

> "Would a trained coder hesitate between these?"

If YES:

1. Try sharpening definitions
2. If still ambiguous → merge

---

## STEP 6 — ONE-SENTENCE COVERAGE TEST

Each code must pass:

> Can I explain what this covers in ONE sentence without listing multiple unrelated things?

If NO → split

---

## STEP 7 — NON-REDUNDANCY KILL STEP

For each code:

"If I delete this, do I lose meaning?"

If NO → delete

---

## STEP 8 — FINAL DIAGNOSTIC UNIQUENESS CHECK

Each code must complete:

> "{code_diagnostic}"

If two codes produce similar completions → merge

---

## HARD RULES

### 1. DOMAIN AWARENESS
Codes from DIFFERENT domains that share similar names may represent DIFFERENT phenomena. Do NOT merge codes across domains unless they are truly identical in meaning.

### 2. NO DOUBLE-BARREL CODES
If a code name contains "and" joining unrelated concepts → split into separate codes.

### 3. NO CAUSE + ATTRIBUTE MIX
Do not combine a cause/reason with a descriptive attribute in a single code. Split into separate codes for each mechanism.

---

## VALIDATION CHECKLIST (USE BEFORE FINALIZING)

Run this on every code:

* [ ] Single valence only
* [ ] Answers ONE question
* [ ] Cannot co-occur with same-valence code
* [ ] Mechanism is pure
* [ ] One-sentence coverage
* [ ] Diagnostic is unique

---

## Code Template

**code_name**
→ 3–5 word noun phrase
→ must reflect ONE dimension only

**definition**
→ clear, interpretive claim
→ must specify what makes this DISTINCT

**diagnostic_test**
→ Must follow: "{code_diagnostic}"
→ Must NOT overlap with any other code

**valence**
→ positive / negative / neutral

**typical_indicators**
→ concrete phrases (not abstract labels)

**source_attributes**
→ all merged origins

---

## Template variables reference

| Variable | Source |
|----------|--------|
| `{domain_diagnostic}` | `dimension_def.prompt_rules.domain_diagnostic` |
| `{code_diagnostic}` | `dimension_def.prompt_rules.code_diagnostic` (NEW field added to PromptRules) |
