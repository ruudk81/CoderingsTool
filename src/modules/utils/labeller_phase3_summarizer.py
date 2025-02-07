import asyncio
from typing import List, Dict
import logging
from tqdm.asyncio import tqdm

try:
    # When running as a script
    from labeller import (
        LabellerConfig, HierarchicalStructure, ThemeSummary
    )
except ImportError:
    # When imported as a module
    from .labeller import (
        LabellerConfig, HierarchicalStructure, ThemeSummary
    )
from prompts import HIERARCHICAL_THEME_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class Phase4Summarizer:
    """Phase 4: Generate summaries for themes"""
    
    def __init__(self, config: LabellerConfig, client):
        self.config = config
        self.client = client
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    async def generate_summaries(self,
                               hierarchy: HierarchicalStructure,
                               var_lab: str) -> List[ThemeSummary]:
        """Generate summaries for each theme explaining how it addresses the research question"""
        logger.info("Phase 4: Generating theme summaries...")
        
        # Create tasks for each theme
        tasks = []
        for theme in hierarchy.themes:
            task = self._generate_theme_summary(theme, hierarchy, var_lab)
            tasks.append(task)
        
        # Execute all tasks with progress bar
        summaries = []
        with tqdm(total=len(tasks), desc="Generating summaries") as pbar:
            for coro in asyncio.as_completed(tasks):
                summary = await coro
                summaries.append(summary)
                pbar.update(1)
        
        return summaries
    
    async def _generate_theme_summary(self,
                                    theme,
                                    hierarchy: HierarchicalStructure,
                                    var_lab: str) -> ThemeSummary:
        """Generate summary for a single theme"""
        async with self.semaphore:
            try:
                # Collect information about the theme
                theme_info = self._collect_theme_info(theme, hierarchy)
                
                # Create prompt
                prompt = self._create_summary_prompt(theme, theme_info, var_lab)
                
                # Get summary from LLM
                response = await self._get_llm_response(prompt)
                
                return ThemeSummary(
                    theme_id=theme.node_id,
                    theme_label=theme.label,
                    summary=response.get("summary", ""),
                    relevance_to_question=response.get("relevance", "")
                )
                
            except Exception as e:
                logger.error(f"Error generating summary for theme {theme.node_id}: {e}")
                return ThemeSummary(
                    theme_id=theme.node_id,
                    theme_label=theme.label,
                    summary=f"Error generating summary: {str(e)}",
                    relevance_to_question="Unable to analyze relevance due to error"
                )
    
    def _collect_theme_info(self, theme, hierarchy: HierarchicalStructure) -> Dict:
        """Collect comprehensive information about a theme"""
        info = {
            "theme_label": theme.label,
            "topics": [],
            "total_clusters": 0,
            "representative_codes": [],
            "representative_descriptions": []
        }
        
        # Collect information from all topics and codes
        code_frequency = {}
        desc_samples = []
        
        for topic in theme.children:
            topic_info = {
                "label": topic.label,
                "codes": []
            }
            
            for code in topic.children:
                topic_info["codes"].append(code.label)
                info["total_clusters"] += len(code.cluster_ids)
                
                # Count code frequencies (simplified for this implementation)
                code_frequency[code.label] = code_frequency.get(code.label, 0) + 1
            
            info["topics"].append(topic_info)
        
        # Get most frequent codes
        sorted_codes = sorted(code_frequency.items(), key=lambda x: x[1], reverse=True)
        info["representative_codes"] = [code for code, _ in sorted_codes[:10]]
        
        return info
    
    def _create_summary_prompt(self, theme, theme_info: Dict, var_lab: str) -> str:
        """Create prompt for theme summary"""
        prompt = HIERARCHICAL_THEME_SUMMARY_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{theme_label}", theme.label)
        prompt = prompt.replace("{language}", self.config.language)
        
        # Add theme structure information
        structure_text = f"\nTheme: {theme_info['theme_label']}"
        structure_text += f"\nTotal response clusters: {theme_info['total_clusters']}"
        structure_text += "\n\nTopics within this theme:"
        
        for topic in theme_info['topics']:
            structure_text += f"\n- {topic['label']}"
            structure_text += f"\n  Codes: {', '.join(topic['codes'][:5])}"
            if len(topic['codes']) > 5:
                structure_text += f" (and {len(topic['codes']) - 5} more)"
        
        structure_text += "\n\nMost representative codes across the theme:"
        for code in theme_info['representative_codes'][:10]:
            structure_text += f"\n- {code}"
        
        prompt = prompt.replace("{theme_structure}", structure_text)
        
        return prompt
    
    async def _get_llm_response(self, prompt: str) -> Dict:
        """Get response from LLM with retry logic"""
        messages = [
            {"role": "system", "content": "You are an expert in qualitative data analysis and thematic summarization."},
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
                    max_tokens=1000,
                    response_format={ "type": "json_object" }
                )
                
                # Parse JSON response
                content = response.choices[0].message.content
                return self._parse_json_response(content)
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def _parse_json_response(self, content: str) -> Dict:
        """Parse JSON response from LLM"""
        import json
        
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "summary": "Unable to generate summary",
                "relevance": "Unable to analyze relevance"
            }


if __name__ == "__main__":
    """Test Phase 4 with actual cached data from Phase 3"""
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
    
    # Load phase 3 results from cache
    phase3_data = cache_manager.load_intermediate_data(filename, "phase3_hierarchy", dict)
    
    if phase3_data:
        print("Loaded phase 3 results from cache")
        
        # Extract components
        hierarchy = phase3_data['hierarchy']
        merged_clusters = phase3_data['merged_clusters']
        
        print(f"Found {len(hierarchy.themes)} themes")
        
        # Get variable label
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Initialize configuration
        config = LabellerConfig(
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODEL
        )
    
        # Initialize phase 4 summarizer
        client = instructor.from_openai(AsyncOpenAI(api_key=config.api_key))
        phase4 = Phase4Summarizer(config, client)
        
        async def run_test():
            """Run the test"""
            print("=== Testing Phase 4: Theme Summarization with Real Data ===")
            print(f"Variable label: {var_lab}")
            print(f"Number of themes: {len(hierarchy.themes)}")
            
            try:
                # Test individual theme info collection
                for theme in hierarchy.themes[:3]:  # Just test first 3
                    info = phase4._collect_theme_info(theme, hierarchy)
                    print(f"\nTheme {theme.node_id} info:")
                    print(f"  Label: {info['theme_label']}")
                    print(f"  Topics: {len(info['topics'])}")
                    print(f"  Total clusters: {info['total_clusters']}")
                
                # Generate summaries
                summaries = await phase4.generate_summaries(hierarchy, var_lab)
                
                # Save to cache for complete pipeline
                cache_key = 'phase4_summaries'
                cache_manager.cache_intermediate_data(summaries, filename, cache_key)
                print(f"\nSaved summaries to cache with key '{cache_key}'")
                
                # Display results
                print("\n=== Theme Summaries ===")
                for summary in summaries:
                    print(f"\nTHEME {summary.theme_id}: {summary.theme_label}")
                    print(f"\nSummary:")
                    print(summary.summary[:500] + "..." if len(summary.summary) > 500 else summary.summary)
                    print(f"\nRelevance to research question:")
                    print(summary.relevance_to_question[:300] + "..." if len(summary.relevance_to_question) > 300 else summary.relevance_to_question)
                    print("-" * 50)
                
                # Save results
                output_data = [
                    {
                        "theme_id": summary.theme_id,
                        "theme_label": summary.theme_label,
                        "summary": summary.summary,
                        "relevance_to_question": summary.relevance_to_question
                    }
                    for summary in summaries
                ]
                
                output_file = Path("phase4_test_results.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"\nResults saved to: {output_file}")
                
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
        print("  4. python phase3_organizer.py")