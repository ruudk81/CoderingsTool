# Codebook Quality Investigation

## Problem Statement

Pipeline run (Merk X, n=2000) produced:
- **10 codes + 1 catchall** — too abstract, not concrete enough
- **383 ideas (21%) in "overig/anders"** — far too many
- **Extreme imbalance**: Code 1 = 734 (40%), Code 6 = 6 (0.3%)
- **Attribute fragmentation**: dominant (207, 188) alongside single-digit attributes

Pipeline funnel: 1852 ideas → 6 domains → 32 facets → 68 attributes → 44 candidate codes → **10 final codes**

## Root Cause Hypotheses

| # | Hypothesis | Likely? | Evidence |
|---|-----------|---------|----------|
| H1 | P4.5 consolidation prompt too aggressive | **HIGH** | "MAXIMAL REDUCTION" as Principle #1; 44→10 = 77% reduction |
| H2 | Cross-domain overlap creates false merges | **HIGH** | "Betrouwbaarheid" appears in 3+ domains, merged into 1 code |
| H3 | No frequency data in consolidation | **HIGH** | LLM can't see that merging loses 300+ ideas |
| H4 | Embedding pre-filter excludes correct codes | **MEDIUM** | 10→5 codes, some best matches may be excluded |
| H5 | Domain size imbalance biases code generation | **LOW** | Sustainability=705 ideas vs Other=13 |

## Investigation Steps

### Step 1: Map the 44→10 consolidation
- [x] **Status**: COMPLETED

#### Findings

The 44 candidates came from 4 domains (no domain provenance tags — only valence tags were present):

| Final code | Valence | Merged from | Count | Problem |
|---|---|---|---|---|
| Duurzaamheid en ethiek (+) | + | C1, C2, C3 | 3 merged | C1=sustainability, C2=ethics, C3=social impact — **3 distinct phenomena collapsed** |
| Duurzaamheid en ethiek (-) | - | C4, C5 | 2 merged | OK — both are negative sustainability |
| Financiële dienstverlening (+) | + | C6, C7, C8, C9, C10 | 5 merged | Products, stability, digital, pricing, general — **too diverse** |
| Financiële dienstverlening (-) | - | C11, C12, C13 | 3 merged | Costs, restrictions, stability — reasonable |
| Klantgerichtheid en service (+) | + | C14, C15, C16, C17, C18, C19, **C32, C38** | **8 merged** | Trust, transparency, access, personal service, expertise, satisfaction + **cross-domain** C32 (brand trust) and C38 (brand expertise) — **massive over-merge, includes cross-domain codes** |
| Klantgerichtheid en service (-) | - | C20-C25 | 6 merged | All negative service — reasonable |
| Merkimago en symboliek (+) | + | C26, C28, C29, C31, C33, C34, C35, C36, C37 | **9 merged** | Symbols, recognition, advertising, brand image, innovation, identity, personality, accessibility, **market size** — **catch-all for everything brand** |
| Merkimago en symboliek (-) | - | C39, C40, C41 | 3 merged | Old-fashioned, undifferentiated, disconnected — reasonable |
| Merkbekendheid en ervaring (~) | ~ | C27, C30, C42, C44 | 4 merged | Low awareness + no associations — reasonable |
| Vergelijking met andere banken (~) | ~ | C43 | 1 | Only survivor — kept because unique |

#### Key observations

1. **Code 5 (Klantgerichtheid +) absorbed cross-domain codes**: C32 (brand trust) and C38 (brand expertise) were from the brand identity domain but got merged into customer service. This is a **false cross-domain merge**.

2. **Code 7 (Merkimago +) is a catch-all**: 9 codes merged including C37 (Grootte en marktpositie) which is about organizational size/structure — totally different from brand symbolism. This is why "Grootte, schaal, marktpositie: 54" dominates the "other" pile — the merged code lost its specificity.

3. **No domain provenance in the prompt**: The consolidation prompt only showed `(+)` or `(-)` tags, not which domain each code came from. The LLM had no signal that C32 (brand trust) and C14 (service trust) are from different domains and represent different phenomena.

4. **Sustainability mega-code**: C1 (environment), C2 (ethics), C3 (social impact) are genuinely different facets of sustainability — merged into one 734-idea giant.

### Step 2: Analyze the "overig/anders" pile
- [x] **Status**: COMPLETED

#### Domain breakdown of 383 "other" ideas

| Domain | Count | % of other |
|--------|-------|-----------|
| **Merkidentiteit en imago** | **288** | **75%** |
| Financiële producten | 60 | 16% |
| Klantbeleving | 24 | 6% |
| Marketing | 8 | 2% |
| Other | 3 | 1% |

**75% of the "other" pile comes from the brand identity domain.** This is the domain that had 9 candidate codes merged into a single "Merkimago en symboliek (+)" mega-code.

#### Top attributes in "other" pile — what's missing from the codebook

| Attribute | Count | What it represents | Which code should have it? |
|-----------|-------|----|---|
| Merkidentiteit en symboliek | 69 | Logo, eekhoorn, visual identity | Code 7 has this — so **pre-filter or assignment failure** |
| Grootte, schaal, marktpositie | 54 | Small bank, part of Volksbank, market position | **NO CODE EXISTS** — C37 was merged away |
| Algemene merkwaardering | 45 | General impressions, "good bank" | Code 7 covers this broadly — **pre-filter or assignment failure** |
| Toegankelijkheid en doelgroep | 31 | For everyone, volksbank, accessible | **NO CODE EXISTS** — C36 was merged away |
| Perceptie standaard bank | 30 | Just a normal bank, nothing special | **NO CODE EXISTS** — lost in merge |
| Merkpersoonlijkheid | 17 | Friendly, sympathetic brand personality | Code 7 covers this — **assignment failure** |
| Traditioneel/ouderwets imago | 11 | Old-fashioned, not modern | Code 8 covers this — **assignment failure** |
| Politieke positionering | 7 | Left-wing, ideological, political | **NO CODE EXISTS** — never had a code for this |

#### Key findings

1. **4 real codebook gaps** — phenomena that existed as candidate codes (C36, C37) or facets but have no final code:
   - Organizational structure / market position (54 ideas)
   - Accessibility / target group positioning (31 ideas)
   - "Just a normal bank" perception (30 ideas)
   - Political/ideological positioning (7 ideas)

2. **Assignment failures** — 69 + 45 + 17 = 131 ideas have attributes that DO match existing codes but were assigned "other" anyway. This suggests the embedding pre-filter is excluding the correct code from the top-5, or the LLM can't match the idea to the overly abstract mega-code.

3. **The 9→1 brand merge is the root cause**: Code 7 absorbed everything brand-related (symbols, recognition, advertising, image, innovation, identity, personality, accessibility, market size). When a code covers everything, it covers nothing — the LLM can't confidently assign an idea about "kleine bank" to a code called "Merkimago en symboliek".

### Steps 3-4: Root cause analysis and false merges
- [x] **Status**: COMPLETED (from Step 1+2 evidence)

#### Confirmed root causes

**Root Cause 1: MAXIMAL REDUCTION instruction is too aggressive (H1 confirmed)**

The prompt says "MAXIMAL REDUCTION" as Principle #1 and "merge all codes that express the same underlying idea." The LLM interpreted this as a mandate to minimize code count, collapsing 44→10. With 44 input codes in a single prompt call, the LLM's instinct is to simplify aggressively.

**Root Cause 2: No domain provenance in consolidation prompt (H2 confirmed)**

Candidate codes only had `(+)` or `(-)` tags. The LLM couldn't see that C14 (Betrouwbaarheid — from customer service domain) and C32 (Betrouwbaarheid — from brand identity domain) represent different phenomena. Result: C32 was merged into the customer service code, losing the "brand trust" dimension.

False cross-domain merges identified:
- C32 (brand trust) → merged into Code 5 (customer service trust)
- C38 (brand expertise) → merged into Code 5 (customer service expertise)
- C36 (brand accessibility) → merged into Code 7 (brand imagery catch-all)
- C37 (market position) → merged into Code 7 (brand imagery catch-all)

**Root Cause 3: No frequency/importance signal (H3 confirmed)**

The LLM had no way to know that C37 (Grootte en marktpositie) covered 54 ideas. It looks like a "detail" code next to the more prominent brand codes, so it got merged. If the prompt showed "C37: covers ~54 ideas", the LLM would have been less likely to merge it away.

**Root Cause 4: 9→1 brand mega-merge**

The brand identity domain produced 9 positive candidate codes (C26, C28, C29, C31, C33, C34, C35, C36, C37). The LLM merged ALL of them into a single "Merkimago en symboliek (+)" code. This single code is too abstract to be useful — it covers everything from logo recognition to market positioning to innovation to brand personality.

#### Hypothesis H4 (embedding pre-filter) — PARTIALLY confirmed

131 ideas in "other" have attributes that DO match existing codes (Merkidentiteit, Algemene merkwaardering, Merkpersoonlijkheid). These should have been assigned to Code 7 or Code 8 but weren't. Two possible explanations:
1. The embedding pre-filter excluded the correct code from the top-5
2. The code definition is so abstract that the LLM couldn't match — "Merkimago en symboliek" doesn't obviously cover "kleine bank" or "normaal"

Both likely contribute. The pre-filter issue is secondary; the primary issue is that the codes are too abstract.

### Step 5: Attribute-to-code coverage analysis
- [x] **Status**: COMPLETED (covered in Step 2 analysis)

The "other" pile analysis already identified orphan attributes. The key gaps are:
- Grootte, schaal, marktpositie (54 ideas) — no code
- Toegankelijkheid en doelgroepgerichtheid (31 ideas) — no code
- Perceptie standaard/gewone bank (30 ideas) — no code
- Politieke positionering (7 ideas) — no code
- 131 ideas with matching attributes assigned to "other" — assignment failure

### Design discussion: Taxonomy hierarchy and valence

#### The taxonomy hierarchy

The taxonomy is structured as: **dimension → domain → facet → attribute → valence**

Valence is the **lowest level** — it's a property of individual ideas/observations, not of domains or facets. A domain like "duurzaamheid en maatschappelijke verantwoordelijkheid" is valence-neutral: it contains both praise ("duurzaam, milieuvriendelijk") and criticism ("greenwashing, niet echt duurzaam").

#### Current approach: forced valence split at P4

Currently, P4 code generation splits each domain into positive and negative buckets (`domain::pos`, `domain::neg`), generating codes separately for each. This means:
- The attribute inventory is pre-filtered by valence before code generation
- P4.5 consolidation then sees parallel code sets that look like duplicates

#### Problem with this approach

The forced valence split at P4 conflates domain provenance with valence direction. When P4.5 sees `(+) Betrouwbaarheid` from customer service and `(+) Betrouwbaarheid` from brand identity, it can't distinguish them — both show the same `(+)` tag. The domain information is discarded.

Worse: because each domain produces separate positive and negative code sets, cross-domain codes with similar names but different meanings (trust-in-service vs trust-as-brand-trait) are indistinguishable from actual duplicates.

#### Recommendation: remove valence split at P4, let codes emerge naturally

**Proposal**: Generate codes per domain **without** valence split. The valence of each code emerges naturally from the attributes it's built from:
- Attributes like "Milieuvriendelijkheid" naturally produce positive codes
- Attributes like "Twijfel over authenticiteit" naturally produce negative codes

The LLM won't merge "praise for sustainability" with "criticism of greenwashing" — they're ontologically different phenomena that produce different codes without forced separation.

**Safeguard**: Add an instruction to P4's prompt: "Generate separate codes for positive and negative phenomena. Do not combine praise and criticism into a single code."

**Benefits**:
1. Domain provenance stays clean in P4.5 — each code comes from ONE domain (e.g., "klantbeleving"), not from an artificial split ("klantbeleving::pos")
2. Codes that share a name across domains remain distinguishable by their domain tag
3. Fewer candidate codes to consolidate (fewer parallel positive/negative duplicates)
4. Valence becomes a code-level property that emerges from content, not an input constraint

**Risk**: Without the forced split, the LLM *might* create valence-neutral codes. Mitigated by:
- P4 prompt instruction to keep positive/negative separate
- P4.5 Principle 2 (valence structure) — already prohibits cross-valence merging
- The diagnostic_test field naturally captures valence direction

#### Impact on Fix 1

With this change, Fix 1 simplifies:
- **Before**: Show `(domain::valence)` tags → `(klantbeleving::+) Betrouwbaarheid`
- **After**: Show `(domain)` tags → `(klantbeleving) Betrouwbaarheid en integriteit`

The valence is visible in the code name and definition, not in the provenance tag. The provenance tag's job is only to signal domain origin.

---

### Step 6: Proposed fixes

#### Fix 0 (NEW): Remove valence split from P4 code generation

**What**: Generate codes per domain without splitting into positive/negative buckets first.
**Why**: The forced split creates artificial parallel code sets and strips domain context. Valence emerges naturally from attribute content.
**Where**: `qualitative_researcher.py` — modify P4 task creation (lines 815-830) to run one task per domain instead of two (pos/neg).
**Impact**: Fewer candidate codes (perhaps ~30 instead of 44), each with clean domain provenance. P4.5 consolidation sees domain tags, not valence tags.
**Safeguard**: Add P4 prompt instruction: "Generate separate codes for positive and negative phenomena."

#### Fix 1: Add domain provenance to consolidation prompt (updated)

**What**: Show `(domain)` tags on each candidate code — just the domain name, no valence in the tag.
**Why**: Prevents false cross-domain merges (C14 from klantbeleving ≠ C32 from merkidentiteit). Valence is visible in code name/definition.
**Where**: `prompts_exp.py` — modify tag construction (lines 1303-1307) to show domain name from `code_provenance`.
**Depends on**: Fix 0 (clean domain provenance without valence split).

#### Fix 2: Replace MAXIMAL REDUCTION with balanced consolidation

**What**: Change Principle 1 from "MAXIMAL REDUCTION" to something like "BALANCED PARSIMONY — aim for the fewest codes that preserve all distinct phenomena. Do not merge codes that represent different aspects of the subject."
**Why**: The current instruction sets a "minimize at all costs" tone that overrides the MECE and valence constraints.
**Where**: `build_codebook_consolidation_prompt()` in `prompts_exp.py`.

#### Fix 3: Add frequency data to candidate codes

**What**: Show approximate idea count per candidate code (from attribute assignments). E.g., `[C37] (+) Grootte en marktpositie (~54 ideas)`.
**Why**: Gives the LLM a signal that merging C37 away loses coverage of 54 ideas. Without this, all codes look equally expendable.
**Where**: `_build_consolidation_codes_block()` — requires passing attribute assignment counts from step 4a to step 5.

#### Fix 4: Add anti-mega-merge constraint

**What**: Add an explicit instruction: "No single code should absorb more than 4-5 candidate codes. If a proposed merge would combine 6+ candidate codes, split into 2-3 more specific codes."
**Why**: Directly prevents the 9→1 brand mega-merge.
**Where**: Add as a new principle in the consolidation prompt.

#### Fix 5: Add minimum distinctness floor

**What**: Add instruction: "If the total number of codes drops below 12, you are likely over-merging. Review your merges and split back any codes that combine clearly different phenomena."
**Why**: Provides a sanity-check floor without being too rigid.
**Where**: Add to the workflow section of the consolidation prompt.

#### Expected outcome

With all 5 fixes:
- 44 candidate codes → ~15-20 final codes (instead of 10)
- "other" pile < 5% (instead of 21%)
- No single code > 25% of ideas (instead of 40%)
- Brand domain maintains 3-4 distinct codes (instead of 1 mega-code)

## Summary of root causes and fixes

| Root cause | Fix | Impact |
|---|---|---|
| Forced valence split creates false duplicates | **Fix 0**: remove valence split from P4 | Fewer candidate codes, clean domain provenance |
| No domain provenance in consolidation | **Fix 1**: add domain tags (domain only, no valence in tag) | Prevents cross-domain false merges |
| MAXIMAL REDUCTION too aggressive | **Fix 2**: balanced parsimony | Fewer merges overall |
| No frequency signal | **Fix 3**: add idea counts | Prevents merging away high-frequency codes |
| 9→1 brand mega-merge | **Fix 4**: anti-mega-merge | Splits brand into 3-4 codes |
| No minimum code floor | **Fix 5**: minimum distinctness | Safety net against over-consolidation |

## Source Files

- Verbose log: `exports/verbose_logs/M000000_..._step4_20260319_063238_generation_assignment.txt`
- Captured prompts: `exports/prompts/step4_classNcoder_Qd1_combined_2000_generation.json`
- P4.5 prompt: `prompts_exp.py` lines 1283-1422
- Pipeline orchestration: `qualitative_researcher.py` lines 799-875
