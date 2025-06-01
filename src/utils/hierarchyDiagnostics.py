"""Diagnostics for ThematicLabeller hierarchy analysis"""

def analyze_hierarchy(labeled_results, codebook):
    """Analyze the hierarchy assignment to understand discrepancies"""
    
    print("\n" + "="*80)
    print("HIERARCHY DIAGNOSTICS")
    print("="*80)
    
    # 1. Analyze codebook structure
    print("\n📋 CODEBOOK STRUCTURE:")
    print(f"  - {len(codebook.themes)} Themes")
    print(f"  - {len(codebook.topics)} Topics")
    print(f"  - {len(codebook.subjects)} Subjects")
    
    # Count clusters in codebook
    codebook_clusters = set()
    for theme in codebook.themes:
        codebook_clusters.update(theme.direct_clusters)
    for topic in codebook.topics:
        codebook_clusters.update(topic.direct_clusters)
    for subject in codebook.subjects:
        codebook_clusters.update(subject.direct_clusters)
    
    print(f"  - {len(codebook_clusters)} clusters assigned in codebook")
    
    # 2. Analyze actual assignments
    print("\n📊 ACTUAL ASSIGNMENTS:")
    theme_assignments = {}
    topic_assignments = {}
    keyword_assignments = {}
    unmapped_count = 0
    other_count = 0
    
    for result in labeled_results:
        if result.response_segment:
            for segment in result.response_segment:
                if segment.Theme:
                    for theme_id, theme_label in segment.Theme.items():
                        if theme_id == 999:
                            if "Unmapped" in theme_label:
                                unmapped_count += 1
                            else:
                                other_count += 1
                        theme_assignments[theme_id] = theme_assignments.get(theme_id, 0) + 1
                
                if segment.Topic:
                    for topic_id in segment.Topic:
                        topic_assignments[topic_id] = topic_assignments.get(topic_id, 0) + 1
                
                if segment.Keyword:
                    for keyword_id in segment.Keyword:
                        keyword_assignments[keyword_id] = keyword_assignments.get(keyword_id, 0) + 1
    
    print(f"  - {len(theme_assignments)} Themes with assignments")
    print(f"  - {len(topic_assignments)} Topics with assignments")
    print(f"  - {len(keyword_assignments)} Keywords with assignments")
    print(f"  - {unmapped_count} unmapped segments")
    print(f"  - {other_count} 'other' segments (below threshold)")
    
    # 3. Show themes in codebook vs assigned
    print("\n🔍 THEME ANALYSIS:")
    print("Themes in codebook:")
    for theme in sorted(codebook.themes, key=lambda x: x.numeric_id):
        count = theme_assignments.get(int(theme.numeric_id), 0)
        print(f"  Theme {theme.id}: {theme.label} - {count} segments assigned")
    
    # 4. Find missing clusters
    print("\n❓ MISSING CLUSTERS:")
    all_cluster_ids = set(keyword_assignments.keys())
    missing_from_codebook = all_cluster_ids - codebook_clusters
    if missing_from_codebook:
        print(f"  Clusters not in codebook: {sorted(missing_from_codebook)}")
    else:
        print("  All clusters are in the codebook")
    
    # 5. Assignment distribution
    print("\n📈 ASSIGNMENT DISTRIBUTION:")
    if theme_assignments:
        sorted_themes = sorted(theme_assignments.items(), key=lambda x: x[1], reverse=True)
        for theme_id, count in sorted_themes[:10]:  # Top 10
            theme_label = "Unknown"
            if theme_id == 999:
                theme_label = "Other/Unmapped"
            else:
                for theme in codebook.themes:
                    if int(theme.numeric_id) == theme_id:
                        theme_label = theme.label
                        break
            print(f"  Theme {theme_id}: {theme_label} - {count} segments")
    
    return {
        'codebook_themes': len(codebook.themes),
        'assigned_themes': len(theme_assignments),
        'codebook_clusters': len(codebook_clusters),
        'assigned_clusters': len(keyword_assignments),
        'unmapped_segments': unmapped_count,
        'other_segments': other_count
    }


def find_unassigned_clusters(labeled_results, original_cluster_count=35):
    """Find which clusters were not assigned"""
    assigned_clusters = set()
    
    for result in labeled_results:
        if result.response_segment:
            for segment in result.response_segment:
                if segment.micro_cluster:
                    cluster_id = list(segment.micro_cluster.keys())[0]
                    assigned_clusters.add(cluster_id)
    
    all_clusters = set(range(original_cluster_count))
    unassigned = all_clusters - assigned_clusters
    
    if unassigned:
        print(f"\n⚠️ Unassigned clusters: {sorted(unassigned)}")
    else:
        print("\n✅ All clusters were assigned")
    
    return unassigned


# Usage example:
if __name__ == "__main__":
    print("Import this module and use:")
    print("  from utils.hierarchyDiagnostics import analyze_hierarchy, find_unassigned_clusters")
    print("  stats = analyze_hierarchy(labeled_results, codebook)")
    print("  unassigned = find_unassigned_clusters(labeled_results)")