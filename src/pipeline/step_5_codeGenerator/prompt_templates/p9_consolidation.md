You are an expert in qualitative research.

Your task is to generate a parsimonious and unambiguous codebook from {{n_raw_codes}} candidate codes.
The codebook must contain codes that are mutually exclusive and collectively exhaustive. A critical aspect is that there is no conceptual overlap between codes, and codes should be semantically unambiguous through the lens of the coding dimension.

<survey_context>
Survey question: "{{survey_question}}"
Language: {{language}}
{{dataset_context_section}}
</survey_context>

<dimension_context>
Dimension: {{dimension_name}} — {{dimension_description}}
</dimension_context>

Each candidate code below shows its idea counts per valence as (+positive / ○neutral / −negative).

<candidate_codes>
{{codes_block}}
</candidate_codes>

Before generating your final codes, you MUST work through your analysis step-by-step in a scratchpad. In your scratchpad field:

<workflow>
## STEP 1 — VALENCE POLICY (prevalence-gated, MANDATORY FIRST PASS)

Valence COLOURS the codebook. Use the per-valence counts `(+positive / ○neutral / −negative)` shown on each candidate code. A pole counts as well-represented when it is at least ~10% of the phenomenon's ideas AND more than a stray few ideas (not one or two).

Decide per phenomenon:
- BOTH poles well-represented → keep TWO codes (one positive, one negative). This is the DEFAULT — opposite evaluations of the same phenomenon MUST NOT be merged into one neutral code.
  Example: reliability with +88 / −17 → the negative pole is 16% → keep "Reliable & solid" (positive) AND "Shaky & unreliable" (negative). Do NOT collapse to a single neutral "reliability" code.
- Only ONE pole well-represented (the other a stray few) → ONE code spanning the whole range, named for the underlying DIMENSION (valence neutral), never for the dominant pole (e.g. "size", not "big").
  Example: recognition with +67 / −4 → the negative pole is 2% → ONE neutral code "Brand recognition" covering well-known↔barely-known.
- EXCEPTION: a sparse pole that is a genuinely DISTINCT phenomenon — a different mechanism, not merely the opposite evaluation (e.g. "hypocritical / greenwashing" is not simply "not sustainable") — may remain its own code despite low volume.

## STEP 2 — AGGRESSIVE MERGING WITHIN CLUSTERS

Within each valence cluster:

Merge until a coder would NEVER hesitate between remaining codes.

Strict Merge Rule: If both can apply to the same sentence → merge

## STEP 3 — MECHANISM PURITY CHECK

For each code, ask: Is this describing:
* a value (e.g., fair, responsible)
* a functional property (e.g., fast, easy to use)
* a perception/judgment (e.g., reliable, outdated)
* a cause/reason (e.g., due to specific actions or policies)

If mixed → SPLIT

## STEP 4 — NEIGHBOR STRESS TEST

For every pair of same-valence codes, ask: "Would a trained coder hesitate between these?"

If YES:
1. Try sharpening definitions
2. If still ambiguous → merge

## STEP 5 — ONE-SENTENCE COVERAGE TEST

Each code must pass: Can I explain what this covers in ONE sentence without listing multiple unrelated things?

If NO → split

## STEP 6 — NON-REDUNDANCY KILL STEP

For each code: "If I delete this, do I lose meaning?"

If NO → DELETE it

## STEP 7 — FINAL DIAGNOSTIC UNIQUENESS CHECK

Each code must complete the sentence:
"{{code_diagnostic}}"

Rules:
* The completion must be specific and distinct
* It must reflect the code's valence policy (a single pole when split, or the full dimension when a dimensional code)

If two codes produce similar completions → MERGE

## STEP 8 — PREVALENCE WEIGHTING & STRUCTURAL BALANCING

Use code frequency to shape the FINAL codebook.

8.1 Core Structure Rule
- High-prevalence codes MUST define the main codebook structure
- The codebook should be built around a small number of dominant phenomena

8.2 Low-Prevalence Constraint
Low-frequency codes MUST NOT become standalone codes unless:
- They represent a clearly distinct phenomenon, AND
- They cannot be meaningfully merged upward

Otherwise, they must be:
- Abstracted into a higher-level code, OR
- Combined into a broader shared category

8.3 Balancing Constraint — Structured Differentiation
- DO NOT collapse everything into a single dominant code
- If multiple distinct high- or mid-prevalence patterns exist: → they MUST remain separate

8.4 Final Check
Ask: "Does this code exist because it is conceptually necessary, or just because it appeared rarely?"

If the latter → merge or remove
</workflow>

<hard_rules>
### NO DOUBLE-BARREL CODES
If a code name contains "and" joining unrelated concepts → abstract to single phenomenon code name

### NO CAUSE + ATTRIBUTE MIX
Do not combine a cause/reason with a descriptive attribute in a single code. Split into separate codes for each mechanism.

### ORTHOGONALITY TEST
For any pair of codes: "Can a single observation plausibly fall under both?"
- Yes → merge
- Doubt → merge
- Only if clearly no → keep separate

### NO HIERARCHY
Codes must not be general vs. specific, or principle vs. application.
If this occurs → merge

### NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals).
If the same underlying principle applies → merge

### PRECEDENCE
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for coding
</hard_rules>

<validation_checklist>
Before finalizing, verify each code passes:
- Valence handled per the policy (split only when each pole qualifies; otherwise a dimensional code)
- Cannot co-occur with same-valence code
- Mechanism is pure
- One-sentence coverage
- Diagnostic is unique
- Prevalence weight rule with balancing constraint
</validation_checklist>

<code_template>
Each code must include:
- **code_name**: 3–5 word noun phrase, must reflect ONE dimension only
- **definition**: clear, interpretive claim — must specify what makes this DISTINCT
- **diagnostic_test**: Must complete: "{{code_diagnostic}}" — must NOT overlap with any other code
- **valence**: positive / negative / neutral (use neutral for a dimensional code that spans the range)
- **typical_indicators**: concrete phrases (not abstract labels)
- **source_attributes**: all merged origins
</code_template>

All output MUST be in {{language}}.

Provide output as valid JSON following the response schema provided.
