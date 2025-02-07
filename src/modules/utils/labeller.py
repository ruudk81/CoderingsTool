import asyncio
import functools
import time
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
import instructor
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Project imports
import models
from config import DEFAULT_LANGUAGE, DEFAULT_MODEL, OPENAI_API_KEY
from modules.utils import qualityFilter
from prompts import CLUSTER_LABELING_PROMPT, RESPONSE_SUMMARY_PROMPT, THEME_SUMMARY_PROMPT

# Patch OpenAI client with instructor for structured output
client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))


class LabellerConfig(BaseModel):
    """Configuration for the Labeller utility"""
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_retries: int = 3
    batch_size: int = 10
    max_tokens: int = 4000
    timeout: int = 30
    max_samples_per_cluster: int = 10
    min_samples_for_labeling: int = 3
    language: str = DEFAULT_LANGUAGE


class ClusterInfo(BaseModel):
    """Internal model for cluster information"""
    cluster_id: int
    cluster_type: str  # "theme", "topic", "code"
    items: List[str] = Field(default_factory=list)
    codes: List[str] = Field(default_factory=list)
    descriptions: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LabelResponse(BaseModel):
    """Response model for LLM labeling"""
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Labeller:
    """
    Generates meaningful labels for hierarchical clusters using LLM
    """
    
    def __init__(
        self,
        input_list: List[models.ClusterModel],
        var_lab: str,
        config: LabellerConfig = None,
        verbose: bool = True
    ):
        """
        Initialize the Labeller
        
        Args:
            input_list: List of ClusterModel objects
            var_lab: Variable label/description for context
            config: Configuration object
            verbose: Print progress information
        """
        self.input_list = input_list
        self.var_lab = var_lab
        self.config = config or LabellerConfig()
        self.verbose = verbose
        
        # Storage for cluster information
        self.theme_clusters: Dict[int, ClusterInfo] = {}
        self.topic_clusters: Dict[int, ClusterInfo] = {}
        self.code_clusters: Dict[int, ClusterInfo] = {}
        
        # Extract cluster information
        self._extract_clusters()
        
        # Use the global client
        self.client = client
        
        if self.verbose:
            print(f"Initialized LLM client with model: {self.config.model}")
    
    def _extract_clusters(self):
        """Extract cluster information from input data"""
        if self.verbose:
            print("Extracting cluster information...")
        
        for response_model in self.input_list:
            if not response_model.response_segment:
                continue
                
            for segment in response_model.response_segment:
                # Extract Theme clusters (meta_cluster)
                if segment.meta_cluster:
                    for cluster_id, keywords in segment.meta_cluster.items():
                        if cluster_id not in self.theme_clusters:
                            self.theme_clusters[cluster_id] = ClusterInfo(
                                cluster_id=cluster_id,
                                cluster_type="theme"
                            )
                        self.theme_clusters[cluster_id].items.append(segment.segment_response)
                        if segment.descriptive_code:
                            self.theme_clusters[cluster_id].codes.append(segment.descriptive_code)
                        if segment.code_description:
                            self.theme_clusters[cluster_id].descriptions.append(segment.code_description)
                
                # Extract Topic clusters (meso_cluster)
                if segment.meso_cluster:
                    for cluster_id, keywords in segment.meso_cluster.items():
                        if cluster_id not in self.topic_clusters:
                            self.topic_clusters[cluster_id] = ClusterInfo(
                                cluster_id=cluster_id,
                                cluster_type="topic"
                            )
                        self.topic_clusters[cluster_id].items.append(segment.segment_response)
                        if segment.descriptive_code:
                            self.topic_clusters[cluster_id].codes.append(segment.descriptive_code)
                        if segment.code_description:
                            self.topic_clusters[cluster_id].descriptions.append(segment.code_description)
                
                # Extract Code clusters (mirco_cluster - note the typo in models.py)
                if segment.mirco_cluster:
                    for cluster_id, keywords in segment.mirco_cluster.items():
                        if cluster_id not in self.code_clusters:
                            self.code_clusters[cluster_id] = ClusterInfo(
                                cluster_id=cluster_id,
                                cluster_type="code"
                            )
                        self.code_clusters[cluster_id].items.append(segment.segment_response)
                        if segment.descriptive_code:
                            self.code_clusters[cluster_id].codes.append(segment.descriptive_code)
                        if segment.code_description:
                            self.code_clusters[cluster_id].descriptions.append(segment.code_description)
        
        if self.verbose:
            print(f"Found {len(self.theme_clusters)} theme clusters")
            print(f"Found {len(self.topic_clusters)} topic clusters")
            print(f"Found {len(self.code_clusters)} code clusters")
    
    def _get_cluster_samples(self, cluster_info: ClusterInfo) -> Dict[str, List[str]]:
        """Get representative samples from a cluster"""
        # Remove duplicates while preserving order
        unique_items = list(dict.fromkeys(cluster_info.items))
        unique_codes = list(dict.fromkeys(cluster_info.codes))
        unique_descriptions = list(dict.fromkeys(cluster_info.descriptions))
        
        # Sample if too many items
        max_samples = self.config.max_samples_per_cluster
        if len(unique_items) > max_samples:
            # Take first, middle, and last items for diversity
            indices = np.linspace(0, len(unique_items) - 1, max_samples, dtype=int)
            unique_items = [unique_items[i] for i in indices]
        
        return {
            "responses": unique_items[:max_samples],
            "codes": unique_codes[:max_samples],
            "descriptions": unique_descriptions[:max_samples]
        }
    
    def _create_prompt(self, cluster_info: ClusterInfo, samples: Dict[str, List[str]]) -> str:
        """Create prompt for labeling a cluster"""
        cluster_type_descriptions = {
            "theme": "overarching theme that encompasses multiple topics",
            "topic": "main topic or category within a theme",
            "code": "specific code or detailed aspect within a topic"
        }
        
        # Format the imported prompt with our variables
        prompt = CLUSTER_LABELING_PROMPT.format(
            language=self.config.language,
            var_lab=self.var_lab,
            cluster_type=cluster_info.cluster_type,
            n_items=len(cluster_info.items),
            cluster_type_description=cluster_type_descriptions[cluster_info.cluster_type],
            responses=chr(10).join(f"- {item}" for item in samples['responses'] if item),
            codes=chr(10).join(f"- {code}" for code in samples['codes'] if code),
            descriptions=chr(10).join(f"- {desc}" for desc in samples['descriptions'] if desc)
        )
        
        return prompt
    
    async def _label_cluster(self, cluster_info: ClusterInfo) -> LabelResponse:
        """Generate label for a single cluster using LLM"""
        samples = self._get_cluster_samples(cluster_info)
        
        # Skip if not enough samples
        if len(samples['responses']) < self.config.min_samples_for_labeling:
            return LabelResponse(
                label=f"{cluster_info.cluster_type.capitalize()} {cluster_info.cluster_id}",
                confidence=0.5,
                reasoning="Insufficient samples for meaningful labeling"
            )
        
        prompt = self._create_prompt(cluster_info, samples)
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                functools.partial(
                    self.client.chat.completions.create,
                    model=self.config.model,
                    response_model=LabelResponse,
                    max_retries=self.config.max_retries,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing and categorizing text data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            )
            return response
        except Exception as e:
            if self.verbose:
                print(f"Error labeling {cluster_info.cluster_type} cluster {cluster_info.cluster_id}: {e}")
            return LabelResponse(
                label=f"{cluster_info.cluster_type.capitalize()} {cluster_info.cluster_id}",
                confidence=0.0,
                reasoning=f"Error during labeling: {str(e)}"
            )
    
    async def _label_clusters_batch(self, clusters: Dict[int, ClusterInfo]) -> Dict[int, LabelResponse]:
        """Label a batch of clusters asynchronously"""
        results = {}
        
        # Create batches
        cluster_items = list(clusters.items())
        batches = [cluster_items[i:i + self.config.batch_size] 
                  for i in range(0, len(cluster_items), self.config.batch_size)]
        
        for batch in batches:
            tasks = [self._label_cluster(cluster_info) for _, cluster_info in batch]
            batch_results = await asyncio.gather(*tasks)
            
            for (cluster_id, _), result in zip(batch, batch_results):
                results[cluster_id] = result
        
        return results
    
    def generate_labels(self) -> None:
        """Generate labels for all clusters"""
        if self.verbose:
            print("\nGenerating cluster labels...")
        
        start_time = time.time()
        
        # Run async labeling
        async def label_all():
            # Label themes
            if self.verbose:
                print(f"\nLabeling {len(self.theme_clusters)} theme clusters...")
            theme_labels = await self._label_clusters_batch(self.theme_clusters)
            
            # Label topics
            if self.verbose:
                print(f"\nLabeling {len(self.topic_clusters)} topic clusters...")
            topic_labels = await self._label_clusters_batch(self.topic_clusters)
            
            # Label codes
            if self.verbose:
                print(f"\nLabeling {len(self.code_clusters)} code clusters...")
            code_labels = await self._label_clusters_batch(self.code_clusters)
            
            return theme_labels, topic_labels, code_labels
        
        # Run the async function
        theme_labels, topic_labels, code_labels = asyncio.run(label_all())
        
        # Store results
        self.theme_labels = theme_labels
        self.topic_labels = topic_labels
        self.code_labels = code_labels
        
        end_time = time.time()
        
        if self.verbose:
            print(f"\nLabeling completed in {end_time - start_time:.2f} seconds")
            
            # Print some examples
            print("\nExample labels:")
            for cluster_id, label_resp in list(theme_labels.items())[:3]:
                print(f"Theme {cluster_id}: '{label_resp.label}' (confidence: {label_resp.confidence:.2f})")
            for cluster_id, label_resp in list(topic_labels.items())[:3]:
                print(f"Topic {cluster_id}: '{label_resp.label}' (confidence: {label_resp.confidence:.2f})")
            for cluster_id, label_resp in list(code_labels.items())[:3]:
                print(f"Code {cluster_id}: '{label_resp.label}' (confidence: {label_resp.confidence:.2f})")
    
    def to_label_model(self) -> List[models.LabelModel]:
        """Convert results to LabelModel format"""
        if not hasattr(self, 'theme_labels'):
            raise ValueError("Must call generate_labels() before converting to model")
        
        # Create a mapping of what labels to apply to each segment
        segment_labels: Dict[Tuple[int, str], Dict[str, Dict[int, str]]] = defaultdict(lambda: {
            "Theme": {},
            "Topic": {},
            "Code": {}
        })
        
        # Process input data to apply labels
        for response_model in self.input_list:
            if not response_model.response_segment:
                continue
            
            for segment in response_model.response_segment:
                segment_key = (response_model.respondent_id, segment.segment_id)
                
                # Apply theme labels
                if segment.meta_cluster:
                    for cluster_id in segment.meta_cluster.keys():
                        if cluster_id in self.theme_labels:
                            segment_labels[segment_key]["Theme"][cluster_id] = self.theme_labels[cluster_id].label
                
                # Apply topic labels
                if segment.meso_cluster:
                    for cluster_id in segment.meso_cluster.keys():
                        if cluster_id in self.topic_labels:
                            segment_labels[segment_key]["Topic"][cluster_id] = self.topic_labels[cluster_id].label
                
                # Apply code labels
                if segment.mirco_cluster:
                    for cluster_id in segment.mirco_cluster.keys():
                        if cluster_id in self.code_labels:
                            segment_labels[segment_key]["Code"][cluster_id] = self.code_labels[cluster_id].label
        
        # Create LabelModel objects
        label_models = []
        
        for response_model in self.input_list:
            # Create label submodels
            label_submodels = []
            
            if response_model.response_segment:
                for segment in response_model.response_segment:
                    segment_key = (response_model.respondent_id, segment.segment_id)
                    labels = segment_labels[segment_key]
                    
                    # Create LabelSubmodel with all original data plus labels
                    label_submodel = models.LabelSubmodel(
                        segment_id=segment.segment_id,
                        segment_response=segment.segment_response,
                        descriptive_code=segment.descriptive_code,
                        code_description=segment.code_description,
                        code_embedding=segment.code_embedding,
                        description_embedding=segment.description_embedding,
                        meta_cluster=segment.meta_cluster,
                        meso_cluster=segment.meso_cluster,
                        mirco_cluster=segment.mirco_cluster,
                        Theme=labels["Theme"] if labels["Theme"] else None,
                        Topic=labels["Topic"] if labels["Topic"] else None,
                        Code=labels["Code"] if labels["Code"] else None
                    )
                    
                    label_submodels.append(label_submodel)
            
            # Create LabelModel
            label_model = models.LabelModel(
                respondent_id=response_model.respondent_id,
                response=response_model.response,
                response_segment=label_submodels if label_submodels else None
            )
            
            label_models.append(label_model)
        
        return label_models
    
    def generate_summary(self, label_models: List[models.LabelModel]) -> List[models.LabelModel]:
        """Generate a summary for each response based on its labels"""
        if self.verbose:
            print("\nGenerating response summaries...")
        
        async def generate_summaries_async():
            tasks = []
            for model in label_models:
                tasks.append(self._generate_response_summary(model))
            
            return await asyncio.gather(*tasks)
        
        # Run async function
        summaries = asyncio.run(generate_summaries_async())
        
        # Apply summaries to models
        for model, summary in zip(label_models, summaries):
            model.summary = summary
        
        return label_models
    
    async def _generate_response_summary(self, label_model: models.LabelModel) -> str:
        """Generate a summary for a single response"""
        if not label_model.response_segment:
            return "No segments to summarize"
        
        # Collect all labels
        themes = set()
        topics = set()
        codes = set()
        
        for segment in label_model.response_segment:
            if segment.Theme:
                themes.update(segment.Theme.values())
            if segment.Topic:
                topics.update(segment.Topic.values())
            if segment.Code:
                codes.update(segment.Code.values())
        
        if not themes and not topics and not codes:
            return "No labels assigned"
        
        # Use LLM to generate summary
        prompt = RESPONSE_SUMMARY_PROMPT.format(
            language=self.config.language,
            var_lab=self.var_lab,
            original_response=label_model.response,
            themes=', '.join(themes) if themes else 'None',
            topics=', '.join(topics) if topics else 'None',
            codes=', '.join(codes) if codes else 'None'
        )
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                functools.partial(
                    self.client.chat.completions.create,
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": f"You are a {self.config.language} expert summarizing survey responses."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=200  # Summaries should be short
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"Error generating summary for respondent {label_model.respondent_id}: {e}")
            # Fallback to simple summary
            summary_parts = []
            if themes:
                summary_parts.append(f"Theme: {', '.join(list(themes)[:2])}")
            if topics:
                summary_parts.append(f"Topic: {', '.join(list(topics)[:2])}")
            if codes:
                summary_parts.append(f"Code: {', '.join(list(codes)[:2])}")
            
            return "; ".join(summary_parts) if summary_parts else "No specific labels"
    
    def get_representative_items_for_theme(
        self, 
        theme_id: int, 
        label_models: List[models.LabelModel],
        max_codes: int = 5,
        max_segments_per_code: int = 3
    ) -> Dict:
        """Get most representative codes and segments for a theme using embeddings"""
        
        # 1. Collect all items in this theme
        theme_data = {
            'codes': {},  # code_id -> list of items
            'all_embeddings': []
        }
        
        for model in label_models:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.Theme and theme_id in segment.Theme:
                        # Store code data
                        if segment.Code:
                            for code_id, code_label in segment.Code.items():
                                if code_id not in theme_data['codes']:
                                    theme_data['codes'][code_id] = {
                                        'label': code_label,
                                        'items': [],
                                        'count': 0
                                    }
                                
                                theme_data['codes'][code_id]['items'].append({
                                    'embedding': segment.code_embedding,
                                    'segment': segment.segment_response,
                                    'description': segment.code_description,
                                    'descriptive_code': segment.descriptive_code
                                })
                                theme_data['codes'][code_id]['count'] += 1
                                theme_data['all_embeddings'].append(segment.code_embedding)
        
        if not theme_data['all_embeddings']:
            return {}
        
        # 2. Calculate theme centroid
        theme_centroid = np.mean(theme_data['all_embeddings'], axis=0)
        
        # 3. Find most representative codes
        code_representations = {}
        for code_id, code_data in theme_data['codes'].items():
            # Calculate mean embedding for this code
            code_embedding = np.mean([item['embedding'] for item in code_data['items']], axis=0)
            # Calculate similarity to theme centroid
            similarity = cosine_similarity([code_embedding], [theme_centroid])[0][0]
            code_representations[code_id] = {
                'similarity': similarity,
                'label': code_data['label'],
                'items': code_data['items']
            }
        
        # 4. Select top codes by similarity
        top_codes = sorted(code_representations.items(), 
                          key=lambda x: x[1]['similarity'], 
                          reverse=True)[:max_codes]
        
        # 5. For each top code, find most representative segments
        representative_data = {}
        for code_id, code_data in top_codes:
            code_centroid = np.mean([item['embedding'] for item in code_data['items']], axis=0)
            
            # Calculate similarities for all segments in this code
            similarities = []
            for item in code_data['items']:
                sim = cosine_similarity([item['embedding']], [code_centroid])[0][0]
                similarities.append((sim, item))
            
            # Select top segments
            top_segments = sorted(similarities, key=lambda x: x[0], reverse=True)[:max_segments_per_code]
            
            representative_data[code_id] = {
                'label': code_data['label'],
                'segments': [item[1] for item in top_segments],
                'similarity_to_theme': code_data['similarity'],
                'count': theme_data['codes'][code_id]['count']
            }
        
        return representative_data
    
    async def _generate_theme_summary(self, theme_id: int, theme_label: str, 
                                     representative_items: Dict) -> str:
        """Generate a summary for a single theme using LLM"""
        
        # Format representative items for the prompt
        items_text = ""
        for code_id, code_data in representative_items.items():
            items_text += f"\nMicro cluster: {code_data['label']}\n"
            items_text += f"Relevance to theme: {code_data['similarity_to_theme']:.2f}\n"
            items_text += "Representative examples:\n"
            
            for item in code_data['segments']:
                items_text += f"  - Descriptive code: {item['descriptive_code']}\n"
                items_text += f"    Code description: {item['description']}\n"
        
        prompt = THEME_SUMMARY_PROMPT.format(
            language=self.config.language,
            var_lab=self.var_lab,
            theme_label=theme_label,
            representative_items=items_text
        )
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                functools.partial(
                    self.client.chat.completions.create,
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": f"You are a {self.config.language} expert analyzing survey response themes."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=300  # Theme summaries can be a bit longer
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"Error generating theme summary for theme {theme_id}: {e}")
            return f"Error generating summary for theme {theme_label}"
    
    def generate_theme_summaries(self, label_models: List[models.LabelModel]) -> Dict[int, Dict]:
        """Generate summaries for all themes based on their most representative content"""
        
        if self.verbose:
            print("\nGenerating theme summaries...")
        
        # Extract unique themes
        themes = {}
        for model in label_models:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.Theme:
                        for theme_id, theme_label in segment.Theme.items():
                            themes[theme_id] = theme_label
        
        if self.verbose:
            print(f"Found {len(themes)} unique themes")
        
        # Generate summaries for each theme
        theme_summaries = {}
        
        async def generate_all_theme_summaries():
            tasks = []
            for theme_id, theme_label in themes.items():
                # Get representative items
                representative_items = self.get_representative_items_for_theme(
                    theme_id, label_models
                )
                
                if representative_items:
                    tasks.append((
                        theme_id, 
                        theme_label,
                        representative_items,
                        self._generate_theme_summary(theme_id, theme_label, representative_items)
                    ))
            
            results = []
            for theme_id, theme_label, representative_items, summary_task in tasks:
                summary = await summary_task
                results.append((theme_id, theme_label, representative_items, summary))
            
            return results
        
        # Run async tasks
        summaries = asyncio.run(generate_all_theme_summaries())
        
        # Process results
        for theme_id, theme_label, representative_items, summary in summaries:
            theme_summaries[theme_id] = {
                'label': theme_label,
                'summary': summary,
                'codes': {}
            }
            
            # Add code information
            for code_id, code_data in representative_items.items():
                theme_summaries[theme_id]['codes'][code_id] = {
                    'label': code_data['label'],
                    'count': code_data['count'],
                    'similarity_to_theme': code_data['similarity_to_theme']
                }
        
        return theme_summaries


# Example usage
if __name__ == "__main__":
    """Test the labeller with actual cluster data"""
    import nest_asyncio
    nest_asyncio.apply()
    
    import sys
    from pathlib import Path
    
    # Add project paths
    sys.path.append(str(Path(__file__).parents[2]))  # Add src directory
    
    from cache_manager import CacheManager
    from cache_config import CacheConfig
    import data_io
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load cluster data from cache
    cluster_data = cache_manager.load_from_cache(filename, 'clusters', models.ClusterModel)
    
    if cluster_data:
        print(f"Loaded {len(cluster_data)} cluster results from cache")
        
        # Get variable label from metadata
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Create labeller
        labeller = Labeller(
            input_list=cluster_data,
            var_lab=var_lab,
            verbose=True
        )
        
        # Generate labels
        labeller.generate_labels()
        
        # Convert to label models
        label_models = labeller.to_label_model()
        
        # Generate summaries
        label_models = labeller.generate_summary(label_models)
        
        # Save results to cache
        cache_manager.save_to_cache(label_models, filename, 'labels')
        print(f"Saved {len(label_models)} label results to cache")
        
        # Generate theme summaries
        theme_summaries = labeller.generate_theme_summaries(label_models)
        
        # Print theme summaries
        print("\n\n=== THEME SUMMARIES ===")
        for theme_id, theme_data in sorted(theme_summaries.items()):
            print(f"\nTheme {theme_id}: {theme_data['label']}")
            print(f"Summary: {theme_data['summary']}")
            print("\nCodes within this theme:")
            for code_id, code_info in sorted(theme_data['codes'].items(), 
                                           key=lambda x: x[1]['count'], 
                                           reverse=True):
                print(f"  {code_info['label']} (Count: {code_info['count']}, Similarity: {code_info['similarity_to_theme']:.2f})")
            print("-" * 60)
        
        # Print example results
        print("\n\n=== EXAMPLE RESPONSES ===")
        for model in label_models[:3]:
            print(f"\nRespondent {model.respondent_id}:")
            print(f"Original response: {model.response}")
            print(f"Summary: {model.summary}")
            if model.response_segment:
                for i, segment in enumerate(model.response_segment):
                    print(f"\n  Segment {i+1}:")
                    print(f"    Segment response: {segment.segment_response}")
                    print(f"    Descriptive code: {segment.descriptive_code}")
                    print(f"    Code description: {segment.code_description}")
                    if segment.Theme:
                        print(f"    Themes: {list(segment.Theme.values())}")
                    if segment.Topic:
                        print(f"    Topics: {list(segment.Topic.values())}")
                    if segment.Code:
                        print(f"    Codes: {list(segment.Code.values())}")
    else:
        print("No cluster data found in cache. Please run the clustering step first.")