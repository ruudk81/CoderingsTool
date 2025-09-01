import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import DEFAULT_LANGUAGE

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter


@dataclass
class ClusterAssignmentStats:
    """Statistics for cluster-based assignment"""
    total_ideas: int = 0
    reassigned_to_subclusters: int = 0
    direct_cluster_assignments: int = 0
    parent_cluster_fallbacks: int = 0
    perfect_matches: int = 0
    processing_time_seconds: float = 0.0


class ClusterAssigner:
    """
    Direct cluster-based code assignment with sub-cluster reassignment.
    
    Architecture:
    1. Build cluster-to-codes mapping from enriched codebook
    2. Reassign ideas from parent clusters (12) to sub-clusters (12-1, 12-2)
    3. Direct assignment: idea.cluster -> codes for that cluster
    4. No API calls, no embedding similarity - pure cluster membership
    """
    
    def __init__(
        self,
        cluster_models: List[models.ClusterModel],
        enriched_codebook: List[models.ThemeEnrichedCodebookEntry],
        theme_embeddings: Optional[Dict[str, Any]] = None,
        var_lab: str = "",
        language: str = DEFAULT_LANGUAGE,
        verbose: bool = False):
        
        # Set all instance variables first
        self.cluster_models = cluster_models
        self.enriched_codebook = enriched_codebook
        self.theme_embeddings = theme_embeddings or {}
        self.var_lab = var_lab
        self.language = language
        self.verbose = verbose  # Add this line
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        
        # Results storage
        self._results: List[models.CodeAssignedModel] = []
        self._stats = ClusterAssignmentStats()
        
        # Build mappings once during initialization (after all instance vars are set)
        self._cluster_codes_map = self._build_cluster_codes_mapping()
        self._subcluster_themes_map = self._build_subcluster_themes_mapping()
    
    def _build_cluster_codes_mapping(self) -> Dict[Union[int, str], List[models.ThemeEnrichedCodebookEntry]]:
        """Build mapping from cluster/sub-cluster IDs to their codes"""
        cluster_codes = defaultdict(list)
        
        # Debug: Check what fields are available
        if self.verbose:
            print("\n[DEBUG] Checking enriched_codebook entries:")
            for i, code in enumerate(self.enriched_codebook[:3]):  # Show first 3
                print(f"  Entry {i}: code='{code.code}'")
                print(f"    source_cluster: {getattr(code, 'source_cluster', 'NOT FOUND')}")
                print(f"    theme_cluster_id: {getattr(code, 'theme_cluster_id', 'NOT FOUND')}")
        
        for code in self.enriched_codebook:
            # Use source_cluster as primary cluster association
            if hasattr(code, 'source_cluster') and code.source_cluster is not None:
                cluster_id = str(code.source_cluster)
                cluster_codes[cluster_id].append(code)
            
            # Fallback to theme_cluster_id if no source_cluster
            elif code.theme_cluster_id is not None:
                cluster_id = str(code.theme_cluster_id)
                cluster_codes[cluster_id].append(code)
        
        self.verbose_reporter.stat_line(f"Built cluster-codes mapping for {len(cluster_codes)} clusters")
        for cluster_id, codes in list(cluster_codes.items())[:5]:  # Show first 5
            self.verbose_reporter.stat_line(f"  Cluster {cluster_id}: {len(codes)} codes")
        
        return dict(cluster_codes)
    
    def _build_subcluster_themes_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Build mapping from sub-cluster IDs to their theme information"""
        subcluster_themes = {}
        
        # Extract sub-cluster information from codes' source_cluster field
        parent_to_subclusters = defaultdict(set)
        
        for code in self.enriched_codebook:
            if hasattr(code, 'source_cluster') and code.source_cluster is not None:
                cluster_id = str(code.source_cluster)
                
                # Check if this is a sub-cluster (contains '-')
                if '-' in cluster_id:
                    parent_cluster = cluster_id.split('-')[0]
                    parent_to_subclusters[parent_cluster].add(cluster_id)
                    
                    # Use code's theme information for sub-cluster
                    if cluster_id not in subcluster_themes:
                        subcluster_themes[cluster_id] = {
                            'theme_label': code.theme or '',
                            'theme_description': code.theme_description or '',
                            'codes': []
                        }
                    subcluster_themes[cluster_id]['codes'].append(code.code)
        
        # Debug output
        if self.verbose:
            print(f"\n[DEBUG] Found {len(parent_to_subclusters)} parent clusters with sub-clusters:")
            for parent, subs in list(parent_to_subclusters.items())[:3]:
                print(f"  Cluster {parent} -> {sorted(subs)}")
        
        self.verbose_reporter.stat_line(f"Built sub-cluster themes mapping for {len(subcluster_themes)} clusters")
        
        return subcluster_themes
    
    def _extract_all_ideas(self) -> List[Tuple[str, str, str, Union[int, str], Optional[np.ndarray]]]:
        """Extract all ideas with their cluster assignments"""
        all_ideas = []
        
        for model in self.cluster_models:
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    # Get embedding if available
                    embedding = None
                    if hasattr(idea_submodel, 'idea_embedding') and idea_submodel.idea_embedding is not None:
                        embedding = idea_submodel.idea_embedding
                    
                    all_ideas.append((
                        model.respondent_id,
                        idea_submodel.idea_id,
                        idea_submodel.idea,
                        idea_submodel.initial_cluster,
                        embedding
                    ))
        
        self.verbose_reporter.stat_line(f"Extracted {len(all_ideas)} ideas for cluster assignment")
        return all_ideas
    
    def _reassign_ideas_to_subclusters(self, all_ideas: List[Tuple]) -> List[Tuple]:
        """Reassign ideas from parent clusters (12) to sub-clusters (12-1, 12-2) based on theme similarity"""
        reassigned_ideas = []
        reassignment_count = 0
        
        for respondent_id, idea_id, idea_text, initial_cluster, embedding in all_ideas:
            original_cluster = str(initial_cluster)
            assigned_cluster = original_cluster
            
            # Find sub-clusters for this parent cluster
            sub_clusters = [sc for sc in self._subcluster_themes_map.keys() 
                           if sc.startswith(f"{original_cluster}-")]
            
            if len(sub_clusters) > 1:
                # Multi-theme cluster - need to reassign to best sub-cluster
                if embedding is not None:
                    best_subcluster = self._find_best_subcluster_match(
                        idea_text, embedding, sub_clusters
                    )
                    if best_subcluster != original_cluster:
                        assigned_cluster = best_subcluster
                        reassignment_count += 1
                else:
                    # No embedding available - use first sub-cluster as fallback
                    assigned_cluster = sub_clusters[0]
                    reassignment_count += 1
            
            reassigned_ideas.append((
                respondent_id, idea_id, idea_text, assigned_cluster, embedding
            ))
        
        self._stats.reassigned_to_subclusters = reassignment_count
        self.verbose_reporter.stat_line(f"Reassigned {reassignment_count} ideas to sub-clusters")
        
        return reassigned_ideas
    
    def _find_best_subcluster_match(self, idea_text: str, idea_embedding: np.ndarray, 
                                   sub_clusters: List[str]) -> str:
        """Find the best matching sub-cluster for an idea based on theme similarity"""
        
        best_cluster = sub_clusters[0]  # Default fallback
        best_similarity = -1.0
        
        # Simple text-based matching as fallback if no embeddings
        for sub_cluster in sub_clusters:
            theme_info = self._subcluster_themes_map.get(sub_cluster, {})
            theme_text = theme_info.get('theme_description', '') + ' ' + theme_info.get('theme_label', '')
            
            if theme_text.strip():
                # Simple keyword overlap scoring
                idea_words = set(idea_text.lower().split())
                theme_words = set(theme_text.lower().split())
                overlap = len(idea_words.intersection(theme_words))
                similarity = overlap / (len(idea_words) + len(theme_words) - overlap + 1)  # Jaccard + smoothing
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = sub_cluster
        
        return best_cluster
    
    def _assign_codes_by_cluster(self, ideas: List[Tuple]) -> List[Dict[str, Any]]:
        """Direct assignment: idea.cluster � codes for that cluster"""
        assignment_results = []
        
        for respondent_id, idea_id, idea_text, cluster_id, embedding in ideas:
            cluster_id = str(cluster_id)
            
            # Get codes for this cluster
            available_codes = self._cluster_codes_map.get(cluster_id, [])
            
            if available_codes:
                # Perfect cluster match - assign all codes from cluster
                assigned_codes = [code.code for code in available_codes]
                confidence = 1.0
                rationale = f"Direct cluster assignment from cluster {cluster_id}"
                self._stats.direct_cluster_assignments += 1
                self._stats.perfect_matches += 1
            else:
                # Fallback to parent cluster if sub-cluster has no codes
                parent_cluster = cluster_id.split('-')[0] if '-' in cluster_id else cluster_id
                parent_codes = self._cluster_codes_map.get(parent_cluster, [])
                
                if parent_codes:
                    # Take top 2 codes from parent cluster
                    assigned_codes = [code.code for code in parent_codes[:2]]
                    confidence = 0.8
                    rationale = f"Parent cluster fallback from cluster {parent_cluster}"
                    self._stats.parent_cluster_fallbacks += 1
                else:
                    # Ultimate fallback - take first few codes from any cluster
                    all_codes = [code for codes in self._cluster_codes_map.values() for code in codes]
                    assigned_codes = [code.code for code in all_codes[:2]] if all_codes else ["Unassigned"]
                    confidence = 0.3
                    rationale = f"Global fallback - no codes found for cluster {cluster_id}"
            
            # Get theme assignments from codes
            assigned_themes = []
            for code_name in assigned_codes:
                for code in self.enriched_codebook:
                    if code.code == code_name and code.theme and code.theme not in assigned_themes:
                        assigned_themes.append(code.theme)
            
            assignment_results.append({
                'respondent_id': respondent_id,
                'idea_id': idea_id,
                'idea': idea_text,
                'cluster_id': cluster_id,
                'assigned_codes': assigned_codes,
                'assigned_themes': assigned_themes,
                'assignment_confidence': confidence,
                'assignment_rationale': rationale
            })
        
        return assignment_results
    
    def _merge_results_into_models(self, assignment_results: List[Dict]) -> List[models.CodeAssignedModel]:
        """Merge assignment results back into CodeAssignedModel structure"""
        
        # Group assignments by respondent_id
        respondent_assignments = defaultdict(list)
        for result in assignment_results:
            respondent_assignments[result['respondent_id']].append(result)
        
        coded_models = []
        
        for respondent_id, assignments in respondent_assignments.items():
            # Create AssignedIdeaSubmodel for each idea
            assigned_ideas = []
            for assignment in assignments:
                assigned_idea = models.AssignedIdeaSubmodel(
                    idea_id=assignment['idea_id'],
                    idea=assignment['idea'],
                    initial_cluster=assignment['cluster_id'],  # Now reflects sub-cluster
                    assigned_codes=assignment['assigned_codes'],
                    assigned_themes=assignment['assigned_themes'],
                    assignment_confidence=assignment['assignment_confidence'],
                    assignment_rationale=assignment['assignment_rationale']
                )
                assigned_ideas.append(assigned_idea)
            
            # Create CodeAssignedModel for this respondent
            coded_model = models.CodeAssignedModel(
                respondent_id=respondent_id,
                response='',  # We don't have the full response text in this context
                response_ideas=assigned_ideas,
                assignment_metadata={
                    "assignment_method": "cluster_based_v2",
                    "total_ideas": len(assigned_ideas),
                    "clusters_involved": list(set(a['cluster_id'] for a in assignments)),
                    "assignment_timestamp": time.time()
                }
            )
            coded_models.append(coded_model)
        
        return coded_models
    
    async def assign_codes(self) -> List[models.CodeAssignedModel]:
        """Main method to assign codes using cluster-based direct assignment"""
        start_time = time.time()
        
        self.verbose_reporter.section_header("CLUSTER-BASED CODE ASSIGNMENT v2")
        
        # Step 1: Extract all ideas
        all_ideas = self._extract_all_ideas()
        self._stats.total_ideas = len(all_ideas)
        
        if not all_ideas:
            self.verbose_reporter.stat_line("No ideas found for code assignment")
            return []
        
        self.verbose_reporter.stat_line(f"Processing {len(all_ideas)} ideas with cluster-based assignment")
        self.verbose_reporter.stat_line(f"Available codes across {len(self._cluster_codes_map)} clusters")
        
        # Debug: Check if any sub-clusters were found
        if self.verbose:
            sub_cluster_count = sum(1 for c in self._cluster_codes_map.keys() if '-' in str(c))
            print(f"\n[DEBUG] Cluster breakdown:")
            print(f"  Total clusters with codes: {len(self._cluster_codes_map)}")
            print(f"  Sub-clusters (with '-'): {sub_cluster_count}")
            print(f"  Parent clusters: {len(self._cluster_codes_map) - sub_cluster_count}")
        
        # Step 2: Reassign ideas to sub-clusters
        self.verbose_reporter.step_start("Sub-cluster Reassignment")
        reassigned_ideas = self._reassign_ideas_to_subclusters(all_ideas)
        
        # Step 3: Direct cluster-to-code assignment
        self.verbose_reporter.step_start("Direct Cluster Assignment")
        assignment_results = self._assign_codes_by_cluster(reassigned_ideas)
        
        # Step 4: Merge results back into model structure
        self._results = self._merge_results_into_models(assignment_results)
        
        # Calculate final statistics
        end_time = time.time()
        self._stats.processing_time_seconds = end_time - start_time
        
        # Report comprehensive summary
        self.verbose_reporter.summary("CLUSTER ASSIGNMENT COMPLETED", {
            "Total ideas processed": self._stats.total_ideas,
            "Reassigned to sub-clusters": self._stats.reassigned_to_subclusters,
            "Direct cluster assignments": self._stats.direct_cluster_assignments,
            "Parent cluster fallbacks": self._stats.parent_cluster_fallbacks,
            "Perfect matches": self._stats.perfect_matches,
            "Processing time": f"{self._stats.processing_time_seconds:.2f}s",
            "Assignment method": "cluster_based_v2"
        })
        
        return self._results
    
    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        return asyncio.run(self.assign_codes())
    
    def get_assignment_stats(self) -> ClusterAssignmentStats:
        """Get detailed assignment statistics"""
        return self._stats
    
    def debug_cluster_mapping(self) -> Dict[str, Any]:
        """Debug information about cluster-code mappings"""
        return {
            "cluster_codes_map": {k: len(v) for k, v in self._cluster_codes_map.items()},
            "subcluster_themes_map": list(self._subcluster_themes_map.keys()),
            "total_codes": len(self.enriched_codebook),
            "total_clusters": len(self._cluster_codes_map)
        }