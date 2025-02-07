import asyncio
from typing import List, Dict, Set
import logging
from collections import defaultdict

try:
    # When running as a script
    from labeller import (
        LabellerConfig, MergedCluster, HierarchyNode, HierarchicalStructure
    )
except ImportError:
    # When imported as a module
    from .labeller import (
        LabellerConfig, MergedCluster, HierarchyNode, HierarchicalStructure
    )
from prompts import HIERARCHY_CREATION_PROMPT

logger = logging.getLogger(__name__)


class Phase3Organizer:
    """Phase 3: Create hierarchical organization of clusters"""
    
    def __init__(self, config: LabellerConfig, client):
        self.config = config
        self.client = client
    
    async def create_hierarchy(self,
                             merged_clusters: Dict[int, MergedCluster],
                             var_lab: str) -> HierarchicalStructure:
        """Create 3-level hierarchical structure"""
        logger.info("Phase 3: Creating hierarchical structure...")
        
        # Create themes (level 1)
        themes = await self._create_themes(merged_clusters, var_lab)
        logger.info(f"Created {len(themes)} themes")
        
        # Create topics within themes (level 2)
        for theme in themes:
            topics = await self._create_topics(theme, merged_clusters, var_lab)
            theme.children = topics
            logger.info(f"Created {len(topics)} topics for theme {theme.node_id}")
        
        # Assign codes within topics (level 3)
        cluster_to_path = {}
        for theme in themes:
            for topic in theme.children:
                codes = self._create_codes(topic, merged_clusters)
                topic.children = codes
                
                # Track cluster to path mapping
                for code in codes:
                    for cluster_id in code.cluster_ids:
                        cluster_to_path[cluster_id] = code.node_id
        
        hierarchy = HierarchicalStructure(
            themes=themes,
            cluster_to_path=cluster_to_path
        )
        
        self._validate_hierarchy(hierarchy, merged_clusters)
        
        return hierarchy
    
    async def _create_themes(self,
                           merged_clusters: Dict[int, MergedCluster],
                           var_lab: str) -> List[HierarchyNode]:
        """Create theme-level groupings using LLM"""
        # Prepare cluster information for LLM
        cluster_info = self._prepare_cluster_info(merged_clusters)
        
        # Create prompt
        prompt = self._create_theme_prompt(cluster_info, var_lab)
        
        # Get theme groupings from LLM
        response = await self._get_llm_response(prompt, "themes")
        
        # Convert response to HierarchyNode objects
        themes = []
        for i, theme_data in enumerate(response.get("themes", [])):
            theme_id = str(i + 1)
            theme = HierarchyNode(
                node_id=theme_id,
                level="theme",
                label=theme_data.get("label", f"Theme {theme_id}"),
                children=[],
                cluster_ids=theme_data.get("cluster_ids", [])
            )
            themes.append(theme)
        
        # Ensure all clusters are assigned to a theme
        assigned_clusters = set()
        for theme in themes:
            assigned_clusters.update(theme.cluster_ids)
        
        unassigned = set(merged_clusters.keys()) - assigned_clusters
        if unassigned:
            # Create a theme for unassigned clusters
            other_theme = HierarchyNode(
                node_id=str(len(themes) + 1),
                level="theme",
                label="Other",
                children=[],
                cluster_ids=list(unassigned)
            )
            themes.append(other_theme)
        
        return themes
    
    async def _create_topics(self,
                           theme: HierarchyNode,
                           merged_clusters: Dict[int, MergedCluster],
                           var_lab: str) -> List[HierarchyNode]:
        """Create topic-level groupings within a theme"""
        # If theme has few clusters, create a single topic
        if len(theme.cluster_ids) <= 3:
            topic = HierarchyNode(
                node_id=f"{theme.node_id}.1",
                level="topic",
                label=f"{theme.label} - General",
                children=[],
                cluster_ids=theme.cluster_ids
            )
            return [topic]
        
        # Prepare cluster information for this theme
        theme_clusters = {cid: merged_clusters[cid] for cid in theme.cluster_ids}
        cluster_info = self._prepare_cluster_info(theme_clusters)
        
        # Create prompt
        prompt = self._create_topic_prompt(theme, cluster_info, var_lab)
        
        # Get topic groupings from LLM
        response = await self._get_llm_response(prompt, "topics")
        
        # Convert response to HierarchyNode objects
        topics = []
        for i, topic_data in enumerate(response.get("topics", [])):
            topic_id = f"{theme.node_id}.{i + 1}"
            topic = HierarchyNode(
                node_id=topic_id,
                level="topic",
                label=topic_data.get("label", f"Topic {topic_id}"),
                children=[],
                cluster_ids=topic_data.get("cluster_ids", [])
            )
            topics.append(topic)
        
        # Ensure all theme clusters are assigned to a topic
        assigned_clusters = set()
        for topic in topics:
            assigned_clusters.update(topic.cluster_ids)
        
        unassigned = set(theme.cluster_ids) - assigned_clusters
        if unassigned:
            other_topic = HierarchyNode(
                node_id=f"{theme.node_id}.{len(topics) + 1}",
                level="topic",
                label="Other",
                children=[],
                cluster_ids=list(unassigned)
            )
            topics.append(other_topic)
        
        return topics
    
    def _create_codes(self,
                     topic: HierarchyNode,
                     merged_clusters: Dict[int, MergedCluster]) -> List[HierarchyNode]:
        """Create code-level nodes (one per cluster)"""
        codes = []
        
        for i, cluster_id in enumerate(sorted(topic.cluster_ids)):
            code_id = f"{topic.node_id}.{i + 1}"
            cluster = merged_clusters[cluster_id]
            
            code = HierarchyNode(
                node_id=code_id,
                level="code",
                label=cluster.label,
                children=[],
                cluster_ids=cluster.original_ids  # Use original cluster IDs
            )
            codes.append(code)
        
        return codes
    
    def _prepare_cluster_info(self, clusters: Dict[int, MergedCluster]) -> List[Dict]:
        """Prepare cluster information for LLM prompts"""
        cluster_info = []
        
        for cluster_id, cluster in clusters.items():
            # Get most common codes and descriptions
            code_counts = defaultdict(int)
            desc_counts = defaultdict(int)
            
            for code in cluster.descriptive_codes:
                code_counts[code] += 1
            for desc in cluster.code_descriptions:
                desc_counts[desc] += 1
            
            # Get top 5 most common
            top_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_descs = sorted(desc_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            info = {
                "cluster_id": cluster_id,
                "label": cluster.label,
                "size": len(cluster.descriptive_codes),
                "top_codes": [code for code, _ in top_codes],
                "top_descriptions": [desc for desc, _ in top_descs]
            }
            cluster_info.append(info)
        
        return cluster_info
    
    def _create_theme_prompt(self, cluster_info: List[Dict], var_lab: str) -> str:
        """Create prompt for theme-level grouping"""
        prompt = HIERARCHY_CREATION_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{level}", "theme")
        prompt = prompt.replace("{language}", self.config.language)
        
        # Add cluster information
        clusters_text = []
        for info in cluster_info:
            cluster_text = f"\nCluster {info['cluster_id']}:"
            cluster_text += f"\n- Label: {info['label']}"
            cluster_text += f"\n- Size: {info['size']} items"
            cluster_text += f"\n- Top codes: {', '.join(info['top_codes'][:3])}"
            clusters_text.append(cluster_text)
        
        prompt = prompt.replace("{clusters}", "\n".join(clusters_text))
        
        return prompt
    
    def _create_topic_prompt(self,
                           theme: HierarchyNode,
                           cluster_info: List[Dict],
                           var_lab: str) -> str:
        """Create prompt for topic-level grouping within a theme"""
        prompt = HIERARCHY_CREATION_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{level}", "topic")
        prompt = prompt.replace("{language}", self.config.language)
        
        # Add theme context
        theme_context = f"You are creating topics within the theme: '{theme.label}'"
        prompt = prompt.replace("You are creating", theme_context)
        
        # Add cluster information
        clusters_text = []
        for info in cluster_info:
            cluster_text = f"\nCluster {info['cluster_id']}:"
            cluster_text += f"\n- Label: {info['label']}"
            cluster_text += f"\n- Size: {info['size']} items"
            cluster_text += f"\n- Top codes: {', '.join(info['top_codes'][:3])}"
            clusters_text.append(cluster_text)
        
        prompt = prompt.replace("{clusters}", "\n".join(clusters_text))
        
        return prompt
    
    async def _get_llm_response(self, prompt: str, response_type: str) -> Dict:
        """Get response from LLM with retry logic"""
        messages = [
            {"role": "system", "content": "You are an expert in hierarchical organization and thematic analysis."},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(self.config.max_retries):
            try:
                # Use regular OpenAI client for JSON response
                from openai import AsyncOpenAI
                openai_client = AsyncOpenAI(api_key=self.config.api_key)
                
                response = await openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4000,
                    response_format={ "type": "json_object" }
                )
                
                # Parse JSON response
                content = response.choices[0].message.content
                return self._parse_json_response(content, response_type)
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get LLM response: {e}")
                    # Return fallback response
                    return self._get_fallback_response(response_type)
    
    def _parse_json_response(self, content: str, response_type: str) -> Dict:
        """Parse JSON response from LLM"""
        import json
        
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._get_fallback_response(response_type)
    
    def _get_fallback_response(self, response_type: str) -> Dict:
        """Get fallback response when LLM fails"""
        if response_type == "themes":
            return {"themes": [{"label": "All Clusters", "cluster_ids": []}]}
        elif response_type == "topics":
            return {"topics": [{"label": "All Topics", "cluster_ids": []}]}
        else:
            return {}
    
    def _validate_hierarchy(self,
                          hierarchy: HierarchicalStructure,
                          merged_clusters: Dict[int, MergedCluster]):
        """Validate the hierarchy is complete and consistent"""
        # Check all original clusters are mapped
        all_original_clusters = set()
        for cluster in merged_clusters.values():
            all_original_clusters.update(cluster.original_ids)
        
        mapped_clusters = set(hierarchy.cluster_to_path.keys())
        
        unmapped = all_original_clusters - mapped_clusters
        if unmapped:
            logger.warning(f"Unmapped clusters: {unmapped}")
        
        # Check hierarchy consistency
        for theme in hierarchy.themes:
            theme_clusters = set()
            for topic in theme.children:
                topic_clusters = set()
                for code in topic.children:
                    topic_clusters.update(code.cluster_ids)
                theme_clusters.update(topic_clusters)
            
            # Verify theme cluster_ids match children
            # Note: theme.cluster_ids contains merged IDs, theme_clusters contains original IDs
            # So we skip this validation for now as it's not directly comparable


if __name__ == "__main__":
    """Test Phase 3 with actual cached data from Phase 2"""
    import sys
    from pathlib import Path
    import json
    import pickle
    
    # Add project paths
    sys.path.append(str(Path(__file__).parents[2]))  # Add src directory
    
    from openai import AsyncOpenAI
    import instructor
    from config import OPENAI_API_KEY, DEFAULT_MODEL
    from cache_manager import CacheManager
    from cache_config import CacheConfig
    import data_io
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # File and variable info
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load phase 2 results from cache
    phase2_data = cache_manager.load_intermediate_data(filename, "phase2_merge_mapping", dict)
    
    if phase2_data:
        print("Loaded phase 2 results from cache")
        
        # Extract components
        merge_mapping = phase2_data['merge_mapping']
        cluster_data = phase2_data['cluster_data']
        initial_labels = phase2_data['initial_labels']
        
        # Create merged clusters (from labeller.py apply_merging method)
        from labeller import Labeller
        temp_labeller = Labeller()
        merged_clusters = temp_labeller.apply_merging(cluster_data, initial_labels, merge_mapping)
        print(f"Created {len(merged_clusters)} merged clusters")
        
        # Get variable label
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Initialize configuration
        config = LabellerConfig(
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODEL
        )
    
        # Initialize phase 3 organizer
        client = instructor.from_openai(AsyncOpenAI(api_key=config.api_key))
        phase3 = Phase3Organizer(config, client)
        
        async def run_test():
            """Run the test"""
            print("=== Testing Phase 3: Hierarchical Organization with Real Data ===")
            print(f"Variable label: {var_lab}")
            print(f"Number of merged clusters: {len(merged_clusters)}")
            
            try:
                # Create hierarchy
                hierarchy = await phase3.create_hierarchy(merged_clusters, var_lab)
                
                # Display results
                print("\n=== Hierarchical Structure ===")
                
                for theme in hierarchy.themes:
                    print(f"\nTHEME {theme.node_id}: {theme.label}")
                    print(f"  Clusters: {theme.cluster_ids}")
                    
                    for topic in theme.children:
                        print(f"\n  TOPIC {topic.node_id}: {topic.label}")
                        print(f"    Clusters: {topic.cluster_ids}")
                        
                        for code in topic.children:
                            print(f"\n    CODE {code.node_id}: {code.label}")
                            print(f"      Original clusters: {code.cluster_ids}")
                
                # Save to cache for phase 4
                cache_key = 'phase3_hierarchy'
                cache_data = {
                    'hierarchy': hierarchy,
                    'merged_clusters': merged_clusters
                }
                cache_manager.cache_intermediate_data(cache_data, filename, cache_key)
                print(f"\nSaved results to cache with key '{cache_key}'")
                
                # Show cluster to path mapping (sample)
                print("\n=== Cluster to Path Mapping (first 10) ===")
                for i, (cluster_id, path) in enumerate(sorted(hierarchy.cluster_to_path.items())):
                    if i >= 10:
                        print(f"  ... and {len(hierarchy.cluster_to_path) - 10} more")
                        break
                    print(f"  Cluster {cluster_id} → {path}")
                
                # Save results
                output_data = {
                    "themes": [
                        {
                            "id": theme.node_id,
                            "label": theme.label,
                            "cluster_ids": theme.cluster_ids,
                            "topics": [
                                {
                                    "id": topic.node_id,
                                    "label": topic.label,
                                    "cluster_ids": topic.cluster_ids,
                                    "codes": [
                                        {
                                            "id": code.node_id,
                                            "label": code.label,
                                            "original_clusters": code.cluster_ids
                                        }
                                        for code in topic.children
                                    ]
                                }
                                for topic in theme.children
                            ]
                        }
                        for theme in hierarchy.themes
                    ],
                    "cluster_to_path": hierarchy.cluster_to_path
                }
                
                output_file = Path("phase3_test_results.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"Results saved to: {output_file}")
                
            except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
        
        # Run the test
        asyncio.run(run_test())
    else:
        print("Missing required cached data.")
        print("Please ensure you have run:")
        print("  1. python clusterer.py")
        print("  2. python phase1_labeller.py")
        print("  3. python phase2_merger.py")