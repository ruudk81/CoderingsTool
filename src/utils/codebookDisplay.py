"""Display functions for ThematicLabeller codebook"""

def display_codebook(codebook, title="CODEBOOK"):
    """Display the full hierarchical codebook with all labels and relationships"""
    
    print("\n" + "="*80)
    print(f"📚 {title}")
    print("="*80)
    print(f"Survey Question: {codebook.survey_question}")
    print(f"\nStructure: {len(codebook.themes)} Themes, {len(codebook.topics)} Topics, {len(codebook.subjects)} Subjects")
    print("-"*80)
    
    # Track all assigned clusters
    all_assigned_clusters = set()
    
    # Display themes
    for theme in sorted(codebook.themes, key=lambda x: x.numeric_id):
        print(f"\n🎯 THEME {theme.id}: {theme.label}")
        if theme.description:
            print(f"   Description: {theme.description}")
        if theme.direct_clusters:
            print(f"   Direct clusters: {theme.direct_clusters}")
            all_assigned_clusters.update(theme.direct_clusters)
        
        # Find related topics
        related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
        if related_topics:
            for topic in sorted(related_topics, key=lambda x: x.numeric_id):
                print(f"\n   📍 TOPIC {topic.id}: {topic.label}")
                if topic.description:
                    print(f"      Description: {topic.description}")
                if topic.direct_clusters:
                    print(f"      Direct clusters: {topic.direct_clusters}")
                    all_assigned_clusters.update(topic.direct_clusters)
                
                # Find related subjects
                related_subjects = [s for s in codebook.subjects if s.parent_id == topic.id]
                if related_subjects:
                    for subject in sorted(related_subjects, key=lambda x: x.numeric_id):
                        print(f"\n      🔸 SUBJECT {subject.id}: {subject.label}")
                        if subject.description:
                            print(f"         Description: {subject.description}")
                        if subject.direct_clusters:
                            print(f"         Clusters: {subject.direct_clusters}")
                            all_assigned_clusters.update(subject.direct_clusters)
        else:
            # No topics under this theme
            if not theme.direct_clusters:
                print("   ⚠️  No topics or direct clusters assigned")
    
    # Summary statistics
    print("\n" + "-"*80)
    print(f"📊 SUMMARY:")
    print(f"   Total clusters assigned: {len(all_assigned_clusters)}")
    print(f"   Cluster IDs: {sorted(all_assigned_clusters)}")
    
    # Find themes/topics without assignments
    empty_themes = [t for t in codebook.themes if not t.direct_clusters and 
                    not any(topic.parent_id == t.id for topic in codebook.topics)]
    if empty_themes:
        print(f"\n⚠️  Empty themes: {[t.label for t in empty_themes]}")
    
    return all_assigned_clusters


def display_phase4_changes(codebook_before, final_labels, codebook_after=None):
    """Display what Phase 4 refinement did (or should do)"""
    
    print("\n" + "="*80)
    print("🔄 PHASE 4 REFINEMENT ANALYSIS")
    print("="*80)
    
    # Currently Phase 4 is a simple assignment without LLM refinement
    print("\n📌 Current Phase 4 Implementation:")
    print("   - Takes assignments from Phase 3")
    print("   - Applies probability threshold (0.5)")
    print("   - Does NOT refine labels via LLM")
    print("   - Simply maps clusters to themes/topics/subjects")
    
    print(f"\n📊 Final Labels Generated: {len(final_labels)} clusters")
    
    # Show a sample of final labels
    print("\n📋 Sample Final Label Assignments:")
    for cluster_id in sorted(list(final_labels.keys())[:5]):  # First 5
        label_info = final_labels[cluster_id]
        print(f"\n   Cluster {cluster_id}: {label_info['label']}")
        print(f"   - Theme: {label_info['theme'][0]} (prob: {label_info['theme'][1]:.2f})")
        print(f"   - Topic: {label_info['topic'][0]} (prob: {label_info['topic'][1]:.2f})")
        print(f"   - Subject: {label_info['subject'][0]} (prob: {label_info['subject'][1]:.2f})")
    
    # Check for "other" assignments
    other_themes = sum(1 for v in final_labels.values() if v['theme'][0] == 'other')
    other_topics = sum(1 for v in final_labels.values() if v['topic'][0] == 'other')
    
    print(f"\n⚠️  Assignments to 'other':")
    print(f"   - Themes: {other_themes} clusters")
    print(f"   - Topics: {other_topics} clusters")
    
    print("\n💡 Note: Phase 4 could be enhanced with LLM-based refinement")
    print("   Currently it's just applying the assignments from Phase 3")


def compare_codebooks(codebook1, codebook2, title1="Codebook 1", title2="Codebook 2"):
    """Compare two codebooks to see differences"""
    
    print("\n" + "="*80)
    print(f"🔍 CODEBOOK COMPARISON: {title1} vs {title2}")
    print("="*80)
    
    # Compare basic counts
    print(f"\n📊 Structure Comparison:")
    print(f"   {title1}: {len(codebook1.themes)} themes, {len(codebook1.topics)} topics, {len(codebook1.subjects)} subjects")
    print(f"   {title2}: {len(codebook2.themes)} themes, {len(codebook2.topics)} topics, {len(codebook2.subjects)} subjects")
    
    # Compare theme labels
    themes1 = {t.id: t.label for t in codebook1.themes}
    themes2 = {t.id: t.label for t in codebook2.themes}
    
    if themes1 == themes2:
        print("\n✅ Theme labels are identical")
    else:
        print("\n❌ Theme labels differ:")
        all_ids = set(themes1.keys()) | set(themes2.keys())
        for theme_id in sorted(all_ids):
            label1 = themes1.get(theme_id, "NOT PRESENT")
            label2 = themes2.get(theme_id, "NOT PRESENT")
            if label1 != label2:
                print(f"   Theme {theme_id}: '{label1}' → '{label2}'")


# Usage example
if __name__ == "__main__":
    print("Import this module and use:")
    print("  from utils.codebookDisplay import display_codebook, display_phase4_changes")
    print("  display_codebook(codebook)")
    print("  display_phase4_changes(codebook, final_labels)")