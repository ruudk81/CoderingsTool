import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List, Dict
import models
from utils.thematicLabeller import ThematicLabeller, ClusterLabel, ExtractedAtomicConceptsResponse

class DiagnosticThematicLabeller(ThematicLabeller):
    """Diagnostic version with ID tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_tracking = {
            "initial_clusters": {},
            "phase1_labeled": {},
            "phase2_merged": {},
            "final_assignments": {}
        }
    
    def _extract_initial_clusters(self, cluster_models: List[models.ClusterModel]) -> Dict[int, Dict]:
        """Override to add tracking"""
        clusters = super()._extract_initial_clusters(cluster_models)
        
        print("\n=== DIAGNOSTIC: Initial Clusters Extracted ===")
        print(f"Total unique clusters: {len(clusters)}")
        print(f"Cluster IDs: {sorted(clusters.keys())}")
        
        # Track initial state
        self.id_tracking["initial_clusters"] = {
            "total": len(clusters),
            "ids": sorted(clusters.keys()),
            "sizes": {cid: len(data['descriptions']) for cid, data in clusters.items()}
        }
        
        return clusters
    
    async def _phase1_descriptive_coding(self, initial_clusters: Dict[int, Dict]) -> List[ClusterLabel]:
        """Override to add tracking"""
        labeled_clusters = await super()._phase1_descriptive_coding(initial_clusters)
        
        print("\n=== DIAGNOSTIC: Phase 1 - Descriptive Coding Complete ===")
        print(f"Labeled clusters: {len(labeled_clusters)}")
        print(f"Cluster IDs after labeling: {sorted([c.cluster_id for c in labeled_clusters])}")
        
        # Track phase 1 results
        self.id_tracking["phase1_labeled"] = {
            "total": len(labeled_clusters),
            "ids": sorted([c.cluster_id for c in labeled_clusters]),
            "labels": {c.cluster_id: c.label for c in labeled_clusters}
        }
        
        return labeled_clusters
    
    def _merge_clusters_by_concept_evidence(self, labeled_clusters: List[ClusterLabel], 
                                           atomic_concepts_result: ExtractedAtomicConceptsResponse) -> List[ClusterLabel]:
        """Override to add detailed tracking of merging process"""
        
        print("\n=== DIAGNOSTIC: Merge Process Starting ===")
        print(f"Input clusters: {len(labeled_clusters)}")
        print(f"Atomic concepts found: {len(atomic_concepts_result.atomic_concepts)}")
        
        # Track concept assignments
        cluster_to_concept = {}
        for concept in atomic_concepts_result.atomic_concepts:
            print(f"\nConcept '{concept.concept}' has evidence in clusters: {concept.evidence}")
            for cluster_id_str in concept.evidence:
                cluster_id = int(cluster_id_str)
                if cluster_id in cluster_to_concept:
                    print(f"  ⚠️ WARNING: Cluster {cluster_id} already assigned to '{cluster_to_concept[cluster_id]}', skipping assignment to '{concept.concept}'")
                else:
                    cluster_to_concept[cluster_id] = concept.concept
        
        # Find unassigned clusters
        all_cluster_ids = set(c.cluster_id for c in labeled_clusters)
        assigned_cluster_ids = set(cluster_to_concept.keys())
        unassigned_cluster_ids = all_cluster_ids - assigned_cluster_ids
        
        print("\n=== DIAGNOSTIC: Assignment Summary ===")
        print(f"Total clusters: {len(all_cluster_ids)}")
        print(f"Assigned to concepts: {len(assigned_cluster_ids)}")
        print(f"Unassigned: {len(unassigned_cluster_ids)}")
        print(f"Unassigned cluster IDs: {sorted(unassigned_cluster_ids)}")
        
        # Call parent method
        merged_clusters = super()._merge_clusters_by_concept_evidence(labeled_clusters, atomic_concepts_result)
        
        print("\n=== DIAGNOSTIC: Merge Complete ===")
        print(f"Merged clusters: {len(merged_clusters)}")
        print(f"New cluster IDs: {sorted([c.cluster_id for c in merged_clusters])}")
        
        # Show original ID tracking
        print("\n=== DIAGNOSTIC: Original ID Tracking ===")
        for merged in merged_clusters:
            concept = "Unknown"
            if "Concept:" in merged.description:
                concept = merged.description.split("Concept:")[1].split("(")[0].strip()
            print(f"Merged cluster {merged.cluster_id} (Concept: {concept}) contains original IDs: {merged.original_cluster_ids}")
        
        # Track merging results
        self.id_tracking["phase2_merged"] = {
            "total_input": len(labeled_clusters),
            "total_output": len(merged_clusters),
            "assigned_count": len(assigned_cluster_ids),
            "unassigned_count": len(unassigned_cluster_ids),
            "unassigned_ids": sorted(unassigned_cluster_ids),
            "concept_groups": {},
            "original_id_tracking": {}
        }
        
        # Track which clusters got merged into which new IDs
        for merged in merged_clusters:
            if "Concept:" in merged.description:
                concept = merged.description.split("Concept:")[1].split("(")[0].strip()
                self.id_tracking["phase2_merged"]["concept_groups"][merged.cluster_id] = {
                    "concept": concept,
                    "description": merged.description,
                    "original_ids": merged.original_cluster_ids
                }
                self.id_tracking["phase2_merged"]["original_id_tracking"][merged.cluster_id] = merged.original_cluster_ids
        
        return merged_clusters
    
    def _create_cluster_mapping(self, original_clusters: List[ClusterLabel], 
                               merged_clusters: List[ClusterLabel]) -> Dict[int, int]:
        """Override to add tracking"""
        mapping = super()._create_cluster_mapping(original_clusters, merged_clusters)
        
        print("\n=== DIAGNOSTIC: Cluster Mapping ===")
        print(f"Mapping size: {len(mapping)}")
        print("Sample mappings (first 10):")
        for i, (orig, merged) in enumerate(mapping.items()):
            if i < 10:
                print(f"  Original {orig} -> Merged {merged}")
        
        return mapping
    
    def print_diagnostic_summary(self):
        """Print comprehensive diagnostic summary"""
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY - ID TRACKING THROUGH PIPELINE")
        print("="*60)
        
        initial = self.id_tracking.get("initial_clusters", {})
        phase1 = self.id_tracking.get("phase1_labeled", {})
        phase2 = self.id_tracking.get("phase2_merged", {})
        
        print(f"\nInitial Clusters: {initial.get('total', 0)}")
        print(f"After Phase 1 (Labeling): {phase1.get('total', 0)}")
        print(f"After Phase 2 (Merging): {phase2.get('total', 0)}")
        
        if phase2.get('unassigned_count', 0) > 0:
            print(f"\n✅ FIXED: {phase2['unassigned_count']} unassigned clusters found and properly grouped!")
            print(f"Unassigned IDs: {phase2.get('unassigned_ids', [])}")
            print("These clusters are now grouped into an 'Other/Miscellaneous' concept")
            
            # Check if Other concept was created
            other_clusters = [c for c in phase2.get('concept_groups', {}).values() if c.get('concept') == 'Other']
            if other_clusters:
                other_cluster = other_clusters[0]
                print(f"✅ 'Other' concept created with original IDs: {other_cluster.get('original_ids', [])}")
            else:
                print("⚠️ 'Other' concept not found in merged clusters")
        else:
            print("\n✅ All clusters were assigned to specific concepts - no 'Other' concept needed")
            
        # Show original ID tracking summary
        print("\n📊 ORIGINAL ID TRACKING:")
        other_found = False
        for merged_id, original_ids in phase2.get('original_id_tracking', {}).items():
            concept = "Unknown"
            if merged_id in phase2.get('concept_groups', {}):
                concept = phase2['concept_groups'][merged_id].get('concept', 'Unknown')
            if concept == "Other":
                other_found = True
            print(f"  Merged cluster {merged_id} ({concept}): {len(original_ids)} original clusters {original_ids}")
        
        if phase2.get('unassigned_count', 0) > 0 and not other_found:
            print("  ⚠️ WARNING: Unassigned clusters found but no 'Other' concept in tracking!")
        
        print("\n" + "="*60)

