import asyncio
import time
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

# Project imports
import models
from config import DEFAULT_LANGUAGE
from modules.utils import qualityFilter


class LabellerConfig(BaseModel):
    """Configuration for the Labeller utility"""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    batch_size: int = 10
    timeout: int = 30
    max_samples_per_cluster: int = 10
    min_samples_for_labeling: int = 3


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
        
        # Set up LLM client following qualityFilter pattern
        self._setup_llm()
    
    def _setup_llm(self):
        """Set up the LLM client following qualityFilter pattern"""
        import instructor
        from openai import AsyncOpenAI
        
        self.client = instructor.from_openai(
            AsyncOpenAI(),
            mode=instructor.Mode.JSON
        )
        
        if self.verbose:
            print(f"Initialized LLM client with model: {self.config.model_name}")
    
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
        
        prompt = f"""You are analyzing a cluster of text responses about: {self.var_lab}

This is a {cluster_info.cluster_type} cluster, which represents {cluster_type_descriptions[cluster_info.cluster_type]}.

The cluster contains {len(cluster_info.items)} responses.

Representative response samples:
{chr(10).join(f"- {item}" for item in samples['responses'] if item)}

Associated descriptive codes:
{chr(10).join(f"- {code}" for code in samples['codes'] if code)}

Code descriptions:
{chr(10).join(f"- {desc}" for desc in samples['descriptions'] if desc)}

Based on this content, provide:
1. A concise, descriptive label (2-5 words) that captures the essence of this {cluster_info.cluster_type}
2. Your confidence score (0.0 to 1.0) in this label
3. Brief reasoning explaining why this label best represents the cluster

The label should be specific enough to distinguish this cluster from others, but general enough to encompass all items within it."""
        
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
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing and categorizing text data."},
                    {"role": "user", "content": prompt}
                ],
                response_model=LabelResponse,
                temperature=self.config.temperature,
                max_retries=self.config.max_retries
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
        
        async def generate_summaries():
            tasks = []
            for model in label_models:
                tasks.append(self._generate_response_summary(model))
            
            return await asyncio.gather(*tasks)
        
        summaries = asyncio.run(generate_summaries())
        
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
        
        prompt = f"""Summarize the following response based on its hierarchical labels:

Original response: {label_model.response}

Themes: {', '.join(themes) if themes else 'None'}
Topics: {', '.join(topics) if topics else 'None'}
Codes: {', '.join(codes) if codes else 'None'}

Provide a concise 1-2 sentence summary that captures the main essence of this response based on the labels."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing text based on hierarchical labels."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_retries=self.config.max_retries
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"Error generating summary for respondent {label_model.respondent_id}: {e}")
            return "Error generating summary"


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, r'/workspaces/CoderingsTool/src')
    
    from modules.utils import csvHandler
    
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    csv_handler = csvHandler.CsvHandler()
    
    # Load cluster data
    cluster_data = csv_handler.load_from_csv(filename, 'clusters', models.ClusterModel)
    
    # Create labeller
    labeller = Labeller(
        input_list=cluster_data,
        var_lab="Customer feedback about product quality",
        verbose=True
    )
    
    # Generate labels
    labeller.generate_labels()
    
    # Convert to label models
    label_models = labeller.to_label_model()
    
    # Generate summaries
    label_models = labeller.generate_summary(label_models)
    
    # Save results
    csv_handler.save_to_csv(label_models, filename, 'labels')
    
    # Print example results
    for model in label_models[:3]:
        print(f"\nRespondent {model.respondent_id}:")
        print(f"Summary: {model.summary}")
        if model.response_segment:
            for segment in model.response_segment:
                if segment.Theme:
                    print(f"  Themes: {list(segment.Theme.values())}")
                if segment.Topic:
                    print(f"  Topics: {list(segment.Topic.values())}")
                if segment.Code:
                    print(f"  Codes: {list(segment.Code.values())}")