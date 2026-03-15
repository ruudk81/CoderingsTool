# LLM Prompt Pipeline for Inductive Code Generation

This document contains three reusable prompts for generating qualitative codes from observations using an LLM. The workflow follows a structured inductive coding process:

1. Pattern discovery (cluster observations)
2. Code construction (generate formal codes)
3. Codebook validation (remove overlap and redundancy)

---

# Prompt 1 — Identify Conceptual Clusters PER DOMAIN

You are assisting with qualitative analysis.

The observations below are brand associations.

Dimension: [dynamic insert of name and description]
Domain: [dynamic insert of name and description]

Your task is to identify recurring conceptual patterns in the observations.

Instructions:

* Group observations that express the same underlying idea.
* Focus on meaning, not exact wording.
* Do NOT create formal codes yet.
* Instead, create conceptual clusters and briefly describe the underlying concept.

Output format:

Cluster name
Underlying concept
Example observations from the dataset

Observations:
[insert ladders as observations here]

---

# Prompt 2 — Generate Codes PER DOMAIN

You are creating a qualitative codebook.

Below are conceptual clusters identified from brand associations.

Your task is to convert these clusters into formal codes.

Instructions:

* Each code must represent a distinct concept.
* Codes must be mutually exclusive.
* Avoid codes that are conceptual neighbors.
* Prefer broad but clear concepts rather than many narrow ones.
* Each code must include a short definition.

Output format:

Code name
Definition
Typical indicators (words or phrases that signal the code)

Clusters:
[insert clusters from previous step]

---

# Prompt 3 — Validate and Refine the Codebook PER DOMAIN

You are reviewing a qualitative codebook.

Your task is to evaluate whether the codes are conceptually distinct and suitable for consistent coding.

Instructions:

* Identify codes that overlap conceptually.
* Identify codes that are too similar or redundant.
* Suggest merges or adjustments if necessary.
* Ensure that each code represents a clearly different concept.

Output format:

Evaluation of each code
Potential overlaps
Recommended revisions to the code set

Codebook:
[insert generated codes]


# Prompt 4 — Consolidate codebook entries ACCROSS DOMAINS

You are reviewing codebook entries for different domains

Your task is to evaluate whether the codes are conceptually distinct and suitable for consistent coding.

Instructions:

* Identify codes that overlap conceptually.
* Identify codes that are too similar or redundant.
* Suggest merges or adjustments if necessary.
* Ensure that each code represents a clearly different concept.

Output format:

Evaluation of each code
Potential overlaps
Recommended revisions to the code set

Codebook:
[insert codebook entries]