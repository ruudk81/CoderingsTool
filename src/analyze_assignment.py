#%%

"""
Temporary analysis script to investigate code assignment issues.
Checks alignment between cache steps, assignment logic, and export.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.cacheManager import CacheManager
import models

# Configuration - match your pipeline run
FILENAME = "M000000 Associatiemonitor Merk X net databestand.sav"
VAR_NAME = "Qd1_combined"
SAMPLE_SIZE = 2000
VARIABLE_KEY = f"{VAR_NAME}_{SAMPLE_SIZE}"  # Cache key includes sample size

cache_manager = CacheManager()

print("=" * 80)
print("CODE ASSIGNMENT ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. Load all cached data
# ============================================================================
print("\n[1] LOADING CACHED DATA...")

# Step 3: Ideas extracted (step name: extracted_ideas)
ideas_extracted = cache_manager.load_from_cache(
    FILENAME, "extracted_ideas", VARIABLE_KEY, models.IdeasExtractedModel
)
print(f"  Ideas extracted: {len(ideas_extracted) if ideas_extracted else 0} responses")

# Step 5: Clusters (step name: expanded_clusters)
cluster_results = cache_manager.load_from_cache(
    FILENAME, "expanded_clusters", VARIABLE_KEY, models.ClusterModel
)
print(f"  Cluster results: {len(cluster_results) if cluster_results else 0} responses")

# Step 8: Code assigned (step name: code_assignment_direct)
code_assigned = cache_manager.load_from_cache(
    FILENAME, "code_assignment_direct", VARIABLE_KEY, models.CodeAssignedModel
)
print(f"  Code assigned: {len(code_assigned) if code_assigned else 0} responses")

# Theme enriched codebook (step name: codebook_refinement_enriched)
theme_codebook = cache_manager.load_from_cache(
    FILENAME, "codebook_refinement_enriched", VARIABLE_KEY, models.ThemeEnrichedCodebookModel
)
if theme_codebook and len(theme_codebook) > 0:
    codebook = theme_codebook[0]
    print(f"  Codebook: {len(codebook.codes)} codes")
else:
    codebook = None
    print("  Codebook: NOT FOUND")

# ============================================================================
# 2. Check ID alignment between steps
# ============================================================================
print("\n[2] CHECKING ID ALIGNMENT...")

if ideas_extracted and cluster_results and code_assigned:
    ideas_ids = {r.respondent_id for r in ideas_extracted}
    cluster_ids = {r.respondent_id for r in cluster_results}
    assigned_ids = {r.respondent_id for r in code_assigned}

    print(f"  Ideas extracted IDs: {len(ideas_ids)}")
    print(f"  Cluster results IDs: {len(cluster_ids)}")
    print(f"  Code assigned IDs: {len(assigned_ids)}")

    # Check alignment
    missing_in_cluster = ideas_ids - cluster_ids
    missing_in_assigned = cluster_ids - assigned_ids

    if missing_in_cluster:
        print(f"  ⚠️ {len(missing_in_cluster)} IDs in ideas but not in clusters")
    if missing_in_assigned:
        print(f"  ⚠️ {len(missing_in_assigned)} IDs in clusters but not in assigned")
    if not missing_in_cluster and not missing_in_assigned:
        print("  ✓ All IDs aligned across steps")

# ============================================================================
# 3. Sample: Compare ideas vs assigned codes
# ============================================================================
print("\n[3] SAMPLE COMPARISON: IDEAS vs ASSIGNED CODES")
print("-" * 80)

if ideas_extracted and code_assigned:
    # Build lookup maps
    ideas_by_id = {r.respondent_id: r for r in ideas_extracted}
    assigned_by_id = {r.respondent_id: r for r in code_assigned}
    cluster_by_id = {r.respondent_id: r for r in cluster_results} if cluster_results else {}

    # Show first 10 samples
    sample_ids = list(assigned_by_id.keys())[:10]

    for resp_id in sample_ids:
        assigned = assigned_by_id.get(resp_id)
        ideas = ideas_by_id.get(resp_id)
        cluster = cluster_by_id.get(resp_id)

        print(f"\n{'='*80}")
        print(f"RESPONDENT: {resp_id}")
        print(f"{'='*80}")

        # Original response
        if ideas:
            print(f"\n📝 ORIGINAL RESPONSE:")
            response_text = str(ideas.response) if ideas.response else ""
            print(f"   {response_text[:200]}..." if len(response_text) > 200 else f"   {response_text}")

        # Extracted ideas from Step 3
        if ideas and ideas.response_ideas:
            print(f"\n💡 EXTRACTED IDEAS (Step 3): {len(ideas.response_ideas)} ideas")
            for i, idea in enumerate(ideas.response_ideas, 1):
                print(f"   {i}. [{idea.idea_id}] {idea.idea}")

        # Cluster info
        if cluster and cluster.response_ideas:
            print(f"\n🔗 CLUSTER INFO (Step 5):")
            for idea in cluster.response_ideas:
                print(f"   [{idea.idea_id}] initial_cluster={idea.initial_cluster}, expanded={getattr(idea, 'expanded_cluster', 'N/A')}")

        # Assigned codes from Step 8
        if assigned and assigned.response_ideas:
            print(f"\n✅ ASSIGNED CODES (Step 8): {len(assigned.response_ideas)} assignments")
            for idea in assigned.response_ideas:
                conf = getattr(idea, 'assignment_confidence', 'N/A')
                codes = getattr(idea, 'assigned_codes', None)
                code = codes[0] if codes else 'N/A'
                rationale = getattr(idea, 'assignment_rationale', '')
                rationale_preview = rationale[:100] if rationale else ''
                print(f"   [{idea.idea_id}] → {code} (conf={conf})")
                if rationale_preview:
                    print(f"      Rationale: {rationale_preview}...")

        print()

# ============================================================================
# 4. Check for mismatches between idea_id in different steps
# ============================================================================
print("\n[4] CHECKING IDEA_ID CONSISTENCY...")

if ideas_extracted and code_assigned:
    mismatches = []

    for resp_id in assigned_by_id:
        ideas_record = ideas_by_id.get(resp_id)
        assigned_record = assigned_by_id.get(resp_id)

        if ideas_record and assigned_record:
            ideas_idea_ids = {i.idea_id for i in (ideas_record.response_ideas or [])}
            assigned_idea_ids = {i.idea_id for i in (assigned_record.response_ideas or [])}

            if ideas_idea_ids != assigned_idea_ids:
                mismatches.append({
                    'resp_id': resp_id,
                    'in_ideas_only': ideas_idea_ids - assigned_idea_ids,
                    'in_assigned_only': assigned_idea_ids - ideas_idea_ids
                })

    if mismatches:
        print(f"  ⚠️ {len(mismatches)} respondents have idea_id mismatches!")
        for m in mismatches[:5]:
            print(f"     Respondent {m['resp_id']}:")
            if m['in_ideas_only']:
                print(f"       In ideas but not assigned: {m['in_ideas_only']}")
            if m['in_assigned_only']:
                print(f"       In assigned but not ideas: {m['in_assigned_only']}")
    else:
        print("  ✓ All idea_ids match between steps")

# ============================================================================
# 5. Check code distribution
# ============================================================================
print("\n[5] CODE DISTRIBUTION ANALYSIS...")

if code_assigned:
    from collections import Counter

    code_counts = Counter()
    confidence_by_code = {}

    for record in code_assigned:
        if record.response_ideas:
            for idea in record.response_ideas:
                codes = getattr(idea, 'assigned_codes', None)
                code = codes[0] if codes else 'Unknown'
                conf = getattr(idea, 'assignment_confidence', 0) or 0
                code_counts[code] += 1
                if code not in confidence_by_code:
                    confidence_by_code[code] = []
                confidence_by_code[code].append(conf)

    print(f"\n  Total assignments: {sum(code_counts.values())}")
    print(f"  Unique codes: {len(code_counts)}")

    # Show codes with low average confidence
    print("\n  Codes with lowest average confidence:")
    avg_conf = {code: sum(confs)/len(confs) for code, confs in confidence_by_code.items()}
    for code, avg in sorted(avg_conf.items(), key=lambda x: x[1])[:10]:
        print(f"    {code}: {avg:.2f} avg conf ({code_counts[code]} assignments)")

# ============================================================================
# 6. Show codebook for reference
# ============================================================================
print("\n[6] CODEBOOK REFERENCE...")

if codebook:
    print(f"\n  Available codes ({len(codebook.codes)}):")
    for code in codebook.codes:
        cluster = getattr(code, 'source_cluster', 'N/A')
        print(f"    • {code.code} (cluster: {cluster})")

# ============================================================================
# 6b. Compare Step 6 vs Step 7 codebooks
# ============================================================================
print("\n[6b] STEP 6 vs STEP 7 CODEBOOK COMPARISON...")

# Load Step 6 codebook generation reasoning (has original codes per cluster)
from utils import codeGenerator
codebook_reasoning = cache_manager.load_from_cache(
    FILENAME, "codebook_generation_reasoning", VARIABLE_KEY,
    codeGenerator.CodeGeneratorReasoningResults
)

if codebook_reasoning and len(codebook_reasoning) > 0:
    reasoning = codebook_reasoning[0]
    print(f"  Step 6 codebook reasoning loaded: {len(reasoning.codebook)} codes")

    # Extract cluster IDs from Step 6
    step6_clusters = set()
    step6_code_by_cluster = {}
    for entry in reasoning.codebook:
        cluster_id = entry.get('source_cluster_id', '')
        code_name = entry.get('code', '')
        if cluster_id:
            for c in str(cluster_id).split(','):
                c = c.strip()
                step6_clusters.add(c)
                step6_code_by_cluster[c] = code_name

    print(f"  Step 6 clusters covered: {len(step6_clusters)}")

    # Compare with Step 7 (theme enriched codebook)
    if codebook:
        step7_clusters = set()
        for code in codebook.codes:
            if hasattr(code, 'source_cluster') and code.source_cluster:
                for c in str(code.source_cluster).split(','):
                    step7_clusters.add(c.strip())

        print(f"  Step 7 clusters covered: {len(step7_clusters)}")

        # Find clusters lost in refinement
        lost_in_refinement = step6_clusters - step7_clusters
        if lost_in_refinement:
            print(f"\n  ⚠️ Clusters LOST during Step 7 refinement: {len(lost_in_refinement)}")
            for c in sorted(lost_in_refinement, key=lambda x: int(x.split('-')[0]) if x.replace('-','').isdigit() else 999):
                code = step6_code_by_cluster.get(c, 'Unknown')
                print(f"     Cluster {c}: '{code}'")
        else:
            print("  ✓ No clusters lost during refinement")

else:
    print("  Step 6 codebook reasoning: NOT FOUND")
    step6_clusters = set()
    step6_code_by_cluster = {}

# ============================================================================
# 7. Find clusters without codes
# ============================================================================
print("\n[7] CLUSTERS WITHOUT CODES...")

if cluster_results and codebook:
    # Get all expanded clusters from data
    all_clusters = set()
    cluster_idea_counts = {}
    for record in cluster_results:
        if record.response_ideas:
            for idea in record.response_ideas:
                ec = getattr(idea, 'expanded_cluster', None)
                if ec:
                    all_clusters.add(str(ec))
                    cluster_idea_counts[str(ec)] = cluster_idea_counts.get(str(ec), 0) + 1

    # Get all clusters that have codes
    clusters_with_codes = set()
    for code in codebook.codes:
        if hasattr(code, 'source_cluster') and code.source_cluster:
            for c in str(code.source_cluster).split(','):
                clusters_with_codes.add(c.strip())

    # Find missing
    missing_clusters = all_clusters - clusters_with_codes

    print(f"  Total unique clusters in data: {len(all_clusters)}")
    print(f"  Clusters with codes: {len(clusters_with_codes)}")
    print(f"  Clusters WITHOUT codes: {len(missing_clusters)}")

    if missing_clusters:
        print("\n  ⚠️ Clusters missing codes (with idea counts):")
        for c in sorted(missing_clusters, key=lambda x: cluster_idea_counts.get(x, 0), reverse=True):
            # Check if it was in Step 6
            was_in_step6 = c in step6_clusters
            step6_code = step6_code_by_cluster.get(c, 'N/A')
            status = "LOST in Step7" if was_in_step6 else "NEVER in Step6"
            print(f"     Cluster {c}: {cluster_idea_counts.get(c, 0)} ideas [{status}]")
            if was_in_step6:
                print(f"        Step 6 code was: '{step6_code}'")

# ============================================================================
# 8. Sample ideas from clusters without codes
# ============================================================================
print("\n[8] SAMPLE IDEAS FROM CLUSTERS WITHOUT CODES...")

if cluster_results and missing_clusters:
    samples_shown = 0
    for record in cluster_results:
        if record.response_ideas and samples_shown < 10:
            for idea in record.response_ideas:
                ec = str(getattr(idea, 'expanded_cluster', ''))
                if ec in missing_clusters and samples_shown < 10:
                    print(f"  Cluster {ec}: [{idea.idea_id}] {idea.idea}")
                    samples_shown += 1

# ============================================================================
# 9. Check if missing clusters have embeddings
# ============================================================================
print("\n[9] CHECKING EMBEDDINGS IN MISSING CLUSTERS...")

if cluster_results and missing_clusters:
    for cid in sorted(missing_clusters, key=lambda x: cluster_idea_counts.get(x, 0), reverse=True)[:5]:
        ideas_in_cluster = []
        for record in cluster_results:
            if record.response_ideas:
                for idea in record.response_ideas:
                    ec = str(getattr(idea, 'expanded_cluster', ''))
                    if ec == cid:
                        has_embedding = hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None
                        emb_shape = idea.idea_embedding.shape if has_embedding else None
                        ideas_in_cluster.append({
                            'idea_id': idea.idea_id,
                            'has_embedding': has_embedding,
                            'emb_shape': emb_shape
                        })

        with_emb = sum(1 for i in ideas_in_cluster if i['has_embedding'])
        without_emb = sum(1 for i in ideas_in_cluster if not i['has_embedding'])
        print(f"\n  Cluster {cid}: {len(ideas_in_cluster)} ideas")
        print(f"    With embeddings: {with_emb}")
        print(f"    Without embeddings: {without_emb}")
        if ideas_in_cluster and ideas_in_cluster[0]['has_embedding']:
            print(f"    Sample embedding shape: {ideas_in_cluster[0]['emb_shape']}")

# ============================================================================
# 10. Compare initial_cluster vs expanded_cluster
# ============================================================================
print("\n[10] INITIAL_CLUSTER vs EXPANDED_CLUSTER COMPARISON...")

# Load initial_clusters (Step 5 output / Step 6 input)
initial_clusters_data = cache_manager.load_from_cache(
    FILENAME, "initial_clusters", VARIABLE_KEY, models.ClusterModel
)

if initial_clusters_data:
    print(f"  Loaded initial_clusters: {len(initial_clusters_data)} responses")

    # Get all initial_cluster values
    initial_cluster_ids = set()
    initial_cluster_counts = {}
    for record in initial_clusters_data:
        if record.response_ideas:
            for idea in record.response_ideas:
                ic = getattr(idea, 'initial_cluster', None)
                if ic is not None:
                    ic_str = str(ic)
                    initial_cluster_ids.add(ic_str)
                    initial_cluster_counts[ic_str] = initial_cluster_counts.get(ic_str, 0) + 1

    print(f"  Unique initial_cluster IDs: {len(initial_cluster_ids)}")
    print(f"  Initial cluster IDs: {sorted(initial_cluster_ids, key=lambda x: int(x) if x.lstrip('-').isdigit() else 999)}")

    # Check if missing clusters exist as initial_cluster
    print(f"\n  Checking missing clusters (4, 6, 22, -1) in initial_clusters:")
    for cid in ['4', '6', '22', '-1']:
        count = initial_cluster_counts.get(cid, 0)
        print(f"    Cluster {cid}: {count} ideas in initial_clusters")
else:
    print("  initial_clusters: NOT FOUND in cache")

# Also check expanded_clusters for the same
print(f"\n  Checking missing clusters in expanded_clusters:")
for cid in ['4', '6', '22', '-1']:
    count = cluster_idea_counts.get(cid, 0)
    print(f"    Cluster {cid}: {count} ideas in expanded_clusters")

# ============================================================================
# 11. Check if clusters 4, 6, 22 should have been expanded
# ============================================================================
print("\n[11] CHECKING MULTI-THEME EXPANSION FOR MISSING CLUSTERS...")

# List all expanded_cluster IDs to see sub-clusters
print(f"  All expanded_cluster IDs ({len(all_clusters)}):")
sorted_clusters = sorted(all_clusters, key=lambda x: (int(x.split('-')[0]) if x.split('-')[0].lstrip('-').isdigit() else 999, x))
for c in sorted_clusters:
    count = cluster_idea_counts.get(c, 0)
    # Check if this cluster has a code
    has_code = c in clusters_with_codes
    status = "✓ has code" if has_code else "✗ NO CODE"
    print(f"    {c}: {count} ideas [{status}]")

# Check if 4, 6, 22 have sub-clusters
print("\n  Sub-cluster analysis for missing clusters:")
for base in ['4', '6', '22']:
    sub_clusters = [c for c in all_clusters if c == base or c.startswith(f"{base}-")]
    print(f"    Base cluster {base}: {sub_clusters}")

# ============================================================================
# 12. What clusters ARE in codebook_reasoning.cluster_results?
# ============================================================================
print("\n[12] CLUSTERS IN CODEBOOK_REASONING.CLUSTER_RESULTS...")

if codebook_reasoning and len(codebook_reasoning) > 0:
    reasoning = codebook_reasoning[0]

    if hasattr(reasoning, 'cluster_results') and reasoning.cluster_results:
        cr_cluster_ids = set()
        for cr in reasoning.cluster_results:
            cid = cr.get('cluster_id', '')
            if cid:
                cr_cluster_ids.add(str(cid))

        print(f"  Clusters in cluster_results: {len(cr_cluster_ids)}")
        print(f"  Cluster IDs: {sorted(cr_cluster_ids, key=lambda x: (int(x.split('-')[0]) if x.split('-')[0].lstrip('-').isdigit() else 999, x))}")

        # Check missing clusters
        print(f"\n  Missing clusters (4, 6, 22) in cluster_results?")
        for cid in ['4', '6', '22', '-1']:
            in_cr = cid in cr_cluster_ids
            print(f"    Cluster {cid}: {'YES' if in_cr else 'NO'}")
    else:
        print("  cluster_results: NOT FOUND or empty")

    # Also check step1_summaries (theme extraction output)
    if hasattr(reasoning, 'step1_summaries') and reasoning.step1_summaries:
        s1_cluster_ids = set(str(k) for k in reasoning.step1_summaries.keys())
        print(f"\n  Clusters in step1_summaries (theme extraction): {len(s1_cluster_ids)}")
        print(f"  Missing clusters (4, 6, 22) in step1_summaries?")
        for cid in ['4', '6', '22']:
            in_s1 = cid in s1_cluster_ids
            print(f"    Cluster {cid}: {'YES' if in_s1 else 'NO'}")
    else:
        print("  step1_summaries: NOT FOUND or empty")

# ============================================================================
# 13. Check embeddings in INITIAL_CLUSTERS (Step 6 INPUT)
# ============================================================================
print("\n[13] CHECKING EMBEDDINGS IN INITIAL_CLUSTERS (Step 6 INPUT)...")

if initial_clusters_data:
    for cid in ['4', '6', '22']:
        ideas_in_cluster = []
        for record in initial_clusters_data:
            if record.response_ideas:
                for idea in record.response_ideas:
                    ic = str(getattr(idea, 'initial_cluster', ''))
                    if ic == cid:
                        has_embedding = hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None
                        emb_shape = idea.idea_embedding.shape if has_embedding else None
                        ideas_in_cluster.append({
                            'idea_id': idea.idea_id,
                            'has_embedding': has_embedding,
                            'emb_shape': emb_shape
                        })

        with_emb = sum(1 for i in ideas_in_cluster if i['has_embedding'])
        without_emb = sum(1 for i in ideas_in_cluster if not i['has_embedding'])
        print(f"\n  Cluster {cid} in initial_clusters: {len(ideas_in_cluster)} ideas")
        print(f"    With embeddings: {with_emb}")
        print(f"    Without embeddings: {without_emb}")
        if ideas_in_cluster and ideas_in_cluster[0]['has_embedding']:
            print(f"    Sample embedding shape: {ideas_in_cluster[0]['emb_shape']}")
else:
    print("  initial_clusters not loaded")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
