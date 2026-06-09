# Valence leaking into attributes — diagnosis & plan (Step 4)

**Date:** 2026-06-07
**Status:** IMPLEMENTED (2026-06-07) and verified on Merk X 2000.

> **Implemented:** Both levers are live. (a) The P4/P5/P7 prompts forbid splitting
> attributes by evaluative direction. (b) A post-P7 phase (**P7.5**,
> `valence_consolidator.py`, wired into `run_classifier.py` after P7 / before P8)
> deterministically detects safe valence-split pairs and merges them; the neutral
> merged name/description comes from a small LLM call (`classifier_p7` model,
> cost-tracked as `p7_5_valence_merge`) with a single-token deterministic fallback;
> idea reassignment preserves valence/confidence. `view_valence_split.py` is the
> read-only detector. Decisions taken: grouping = **exact**, merge scope =
> **safe-only** (auto-safe + single-token).
> **Verification run:** the prompt fix prevented the split at the source — P7.5
> reported **0 candidates**, and the taxonomy now carries one neutral
> "Algemene gevoelswaardering" attribute (valence direction in the `valence` field).

---

## 1. Principle

The taxonomy `domain > facet > attribute` must be **descriptive** (*what* is being talked about). **Valence** (`+ / - / 0`) is the **separate evaluative axis**, assigned per idea by P3 (relative to facet) and P6 (relative to attribute). An attribute must **not** encode valence.

## 2. The problem

Step 4 sometimes splits one descriptive concept into two attributes that differ **only in evaluative direction**, baking valence into the attribute name. Observed (Merk X, domain "betrouwbaarheid en waardering", facet "Algemene waardering"):

```
attribute                          +    -    0
Algemene positieve waardering     83    0    2    ← homogeneous +
Algemene niet-positieve waardering 3   16   16    ← homogeneous -/0
```

These are **one** concept ("algemene waardering") split along valence — the `+/-` belongs in the `valence` field, not in two attributes. This double-encodes valence and makes the `valence` field redundant/contradictory.

**Scope: contained.** Only this one facet shows the explicit split; the rest of the taxonomy's attribute names carry no valence lexicon. (The broader evaluativeness — a domain literally named "betrouwbaarheid en waardering", attributes like "Betrouwbaar en degelijk" that are *judgments* — is the **step-3 evaluative-domain** issue, addressed by step 3's descriptive-domain fix (see step 3 `dev/work_to_be_done.md`); this doc is about the literal in-attribute valence split.)

**The prompts already forbid this.** `prompts_classifier.py` says it repeatedly, even in bold: facets/attributes "must be descriptive, not evaluative" (`:288/:507/:790/:1029`), attributes "Be non-evaluative (no judgment, sentiment, or valence)" (`:681-682`), and valence is the separate axis (`:596-600`, `:1110-1111`). So this is a **compliance failure**, not a missing rule.

**Why it still happens.** For **purely-evaluative responses** ("goed", "prima", "niet zo goed") there *is* no descriptive content — the response's entire meaning is its valence. The "descriptive-only" rule has no valid output, so the LLM falls back to splitting by valence. Structural tension: **the pipeline forces an attribute on every idea, but a pure judgment has no descriptive attribute.**

## 3. Deterministic detector recipe

A valence-split artifact is reliably flagged by the **combination** of two deterministic signals (plus same-facet):

1. **Label similarity** — the two attribute names are near-identical (fuzzy / token-set similarity ≥ threshold). E.g. "Algemene **positieve** waardering" ↔ "Algemene **niet-positieve** waardering".
2. **Valence complementarity** — the two attributes skew to **opposite** valence (one mostly `+`, the other mostly `-/0`), computed from the per-idea `valence` field.
3. **Same facet.**

**Why both signals are required** (proven by the data):
- *Label alone over-flags:* "Sparen en spaarrekeningen" / "Betalen en betaalrekeningen" are fuzzy-similar but descriptive → the valence test excludes them (no opposite skew).
- *Valence-homogeneity alone over-flags:* "Concrete natuurbeelden" is 102/1/3 (homogeneous +) but is a fine descriptive attribute → it has **no opposite-label sibling**, so the pairwise rule leaves it alone.

Only the combination isolates the true artifact and leaves real attributes (natuurbeelden, betrouwbaar) untouched.

**Note on the valence signature (correction to a tempting intuition):** within an *artifact* attribute valence is **concentrated** (homogeneous), not spread. *Spread* valence (e.g. "Betrouwbaar en degelijk" = 88/17/2) is the signature of a genuine — or correctly merged — descriptive attribute. So the rule is "near-identical labels + opposite-homogeneous valence", not "valence spread".

**Metric:** token-set similarity (e.g. rapidfuzz `token_set_ratio`) is more robust for multi-word labels than raw char-Levenshtein (which is order/length sensitive). Levenshtein works for the simple case.

## 4. Detection vs merging — the honest line

- **Deterministic detection: yes** (given thresholds). The recipe above flags candidates reproducibly, no LLM.
- **Deterministic auto-merge: only for high-confidence flags** (labels very similar + valence clearly complementary). Broad "differ only in evaluation?" is a **semantic** judgment → wants an LLM/human.
- **Residual limits:** thresholds are config (reproducible but tuned); paraphrased valence-splits with different roots ("Positief beeld" / "Negatieve indruk") have low label-similarity → missed. Prefix tegenpolen ("Betrouwbaar"/"Onbetrouwbaar") *are* caught (high fuzzy + complementary).

## 5. Proposed tooling — `view_valence_split.py`

Read-only, deterministic, no-LLM detector (analogous to `step_6/view_code_divergence.py`). For the dataset in `test_data.py`, load the `taxonomy` cache, and within each facet list attribute **pairs** that are merge-suspects: label-similarity ≥ threshold AND opposite valence skew. Output per candidate: the two names, label-similarity score, each attribute's valence distribution, idea counts, and a sample. Config knobs: `LABEL_SIM_THRESHOLD`, `MIN_SKEW` (how one-sided valence must be), `MIN_COUNT`. Use it to size the problem and to monitor whether prompt/codebook changes reduce it.

## 6. Proposed fix — the real lever is at the source

1. **Primary — P4/P5 prompt (source):** the rule exists but is ignored on pure-evaluation clusters. Add an explicit instruction: *"Never create attributes that differ only in evaluative direction; collapse them into one descriptive attribute and let valence carry the +/-. Responses that are purely a judgment with no descriptive content belong to a single residual 'overall judgment' attribute."* This gives pure-"goed"/"slecht" a single honest home with valence carrying direction.
2. **Safety net — detector (§5)** as monitoring; optionally a high-confidence auto-merge post-pass for the conventional pattern.
3. **Residual category:** treat purely-evaluative ideas as one explicit attribute ("Algemene waardering" / "geen specifieke eigenschap, alleen algemeen oordeel"), valence = direction.

## 7. Open decisions

- Thresholds (`LABEL_SIM_THRESHOLD`, `MIN_SKEW`).
- Auto-merge for high-confidence flags: yes/no.
- Residual-category naming/definition for pure judgments.
- Sequencing vs the step-3 evaluative-domain work (tempering evaluative domains shrinks the "Algemene waardering" bucket; the neutrality rule is step-4-own regardless).

## 8. Blast radius

- Detector (§5): new read-only script — zero risk.
- Source fix (§6.1): P4/P5 prompt edit; re-run step 4 to see effect. Strictly improves downstream (fewer, cleaner attributes → fewer near-duplicate codes in step 5).
- Optional auto-merge post-pass: a step-4 consolidator-style change (deterministic). Step 5/6/7 unaffected by the neutrality fix.
</content>
