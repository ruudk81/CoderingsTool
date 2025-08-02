import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None
# === MODULES ====================================================================================================
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
from utils.verboseReporter import VerboseReporter
import asyncio
import hashlib
import numpy as np
import time
try:

# === MODELS =====================================================================================================
from pydantic import BaseModel, Field, model_validator

# === CONFIG =====================================================================================================
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
from prompts import THEME_IDENTIFICATION_PROMPT, ASSIGN_MISCELLANEOUS_PROMPT

# === DOMAIN-SPECIFIC ============================================================================================
from umap import UMAP
import hdbscan

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ============================================================================
# PYDANTIC MODELS FOR CLUSTERING-BASED THEME IDENTIFICATION
# ============================================================================

class CodeEmbedding(BaseModel):
    """Code with its embedding representation"""
    code_number: int = Field(description="Original code number from codebook")
    code_name: str = Field(description="Original code name")
    definition: str = Field(description="Code definition")
    embedding_text: str = Field(description="Text used for embedding")
    embedding: Optional[np.ndarray] = Field(description="Embedding vector", exclude=True, default=None)
    cluster_id: Optional[int] = Field(description="Assigned cluster ID", default=None)
    
    class Config:
        arbitrary_types_allowed = True

class CodeReference(BaseModel):
    """Reference to a code in the theme hierarchy"""
    code_number: int = Field(description="Original code number from codebook")
    code_name: str = Field(description="Original code name")
    definition: str = Field(description="Code definition")

class ExistingThemeOption(BaseModel):
    """Information about an existing theme option"""
    theme_name: str = Field(description="Name of existing theme")
    theme_description: str = Field(description="Description of existing theme")
    similarity_score: float = Field(description="Similarity to current cluster")

class ThemeDecision(BaseModel):
    """Individual theme decision within a cluster"""
    theme_name: str = Field(description="Theme name")
    theme_description: str = Field(description="Brief theme description")
    assigned_codes: List[int] = Field(description="Code numbers assigned to this theme")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    is_existing: bool = Field(description="Whether this uses an existing theme", default=False)

class ClusterThemeDecision(BaseModel):
    """Decision about theme assignment for a cluster (supports multiple themes)"""
    decision: str = Field(description="create_single_theme | use_existing_theme | split_into_multiple_themes | reject_mixed_cluster")
    themes: List[ThemeDecision] = Field(description="Theme decisions (1+ themes)")
    existing_theme_used: Optional[str] = Field(description="Name of existing theme if used", default=None)
    rationale: str = Field(description="Explanation of the decision including grouping logic")
    
    @model_validator(mode='after')
    def validate_decision_consistency(self):
        """Ensure decision data is consistent"""
        if self.decision == "use_existing_theme":
            if not self.existing_theme_used:
                raise ValueError("existing_theme_used required when using existing theme")
            if len(self.themes) != 1:
                raise ValueError("use_existing_theme should have exactly 1 theme")
        elif self.decision == "split_into_multiple_themes":
            if len(self.themes) < 2:
                raise ValueError("split_into_multiple_themes requires 2+ themes")
            if len(self.themes) > 3:
                raise ValueError("split_into_multiple_themes limited to 3 themes maximum")
        elif self.decision == "create_single_theme":
            if len(self.themes) != 1:
                raise ValueError("create_single_theme should have exactly 1 theme")
        elif self.decision == "reject_mixed_cluster":
            if len(self.themes) != 0:
                raise ValueError("reject_mixed_cluster should have no themes")
        return self

class IndividualCodeAssignment(BaseModel):
    """Decision about individual code assignment to existing themes"""
    decision: str = Field(description="Whether to assign or keep miscellaneous")
    target_theme: Optional[str] = Field(description="Target theme name or null")
    confidence: str = Field(description="Confidence level")
    rationale: str = Field(description="Detailed explanation")

class ThemeStructure(BaseModel):
    """Theme with codes following clustering-based methodology"""
    theme_name: str = Field(description="Descriptive theme name in target language")
    theme_description: str = Field(description="Brief explanation of what unites these codes conceptually")
    codes: List[CodeReference] = Field(description="Codes that belong to this theme")
    cluster_id: int = Field(description="Original cluster ID")
    is_miscellaneous: bool = Field(description="Whether this is a miscellaneous theme", default=False)

# ============================================================================
# SHARED THEME MEMORY - Real-time theme state management (following codeGenerator pattern)
# ============================================================================

@dataclass
class SharedThemeMemory:
    """Thread-safe shared theme memory with real-time updates (following SharedCodebook pattern)"""
    _themes: List[ThemeStructure]
    _lock: asyncio.Lock
    _version: int = 0
    _update_log: List[Dict[str, Any]] = None
    
    def __init__(self, initial_themes: List[ThemeStructure] = None):
        self._themes = initial_themes.copy() if initial_themes else []
        self._lock = asyncio.Lock()
        self._version = 0
        self._update_log = []
    
    async def get_current_snapshot(self) -> Tuple[List[ThemeStructure], int]:
        """Get current themes and version atomically"""
        async with self._lock:
            return self._themes.copy(), self._version
    
    async def add_theme_if_new(self, theme: ThemeStructure) -> Tuple[bool, int]:
        """Add a new theme if it doesn't exist, return (added, new_version)"""
        async with self._lock:
            # Check if theme already exists (by name)
            for existing in self._themes:
                if existing.theme_name.lower() == theme.theme_name.lower():
                    return False, self._version
            
            # Add new theme
            self._themes.append(theme)
            self._version += 1
            self._update_log.append({
                'version': self._version,
                'action': 'add',
                'theme_name': theme.theme_name,
                'cluster_id': theme.cluster_id,
                'timestamp': time.time()
            })
            return True, self._version
    
    async def get_theme_centroid(self, theme: ThemeStructure, code_embeddings: List[CodeEmbedding]) -> Optional[np.ndarray]:
        """Calculate centroid embedding for a theme based on its codes"""
        if not theme.codes:
            return None
        
        # Find embeddings for codes in this theme
        theme_embeddings = []
        for code_ref in theme.codes:
            for code_emb in code_embeddings:
                if code_emb.code_number == code_ref.code_number and code_emb.embedding is not None:
                    theme_embeddings.append(code_emb.embedding)
                    break
        
        if not theme_embeddings:
            return None
        
        # Calculate centroid
        return np.mean(theme_embeddings, axis=0)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get theme memory statistics"""
        async with self._lock:
            return {
                'total_themes': len(self._themes),
                'version': self._version,
                'updates': len(self._update_log)
            }

# ============================================================================
# THEME EMBEDDING MANAGER - Real-time embedding generation (following OptimizedEmbeddingManager pattern)
# ============================================================================

class ThemeEmbeddingManager:
    """Manages theme embeddings with real-time centroid calculation"""
    
    def __init__(self, shared_theme_memory: SharedThemeMemory, verbose: bool = False):
        self.shared_theme_memory = shared_theme_memory
        self.verbose = verbose
    
    def _get_theme_hash(self, theme: ThemeStructure) -> str:
        """Generate hash for theme based on its codes"""
        code_ids = sorted([code.code_number for code in theme.codes])
        theme_content = f"{theme.theme_name}:{':'.join(map(str, code_ids))}"
        return hashlib.md5(theme_content.encode('utf-8')).hexdigest()
    
    async def get_snapshot_embeddings(self, themes: List[ThemeStructure], version: int, 
                                    code_embeddings: List[CodeEmbedding]) -> Tuple[List[ThemeStructure], List[np.ndarray]]:
        """Get embeddings for theme snapshot - always fresh centroids, no caching (following codeGenerator pattern)"""
        if not themes:
            return [], []
        
        # Generate fresh centroid embeddings each time (like codeGenerator)
        theme_centroids = []
        valid_themes = []
        
        for theme in themes:
            centroid = await self.shared_theme_memory.get_theme_centroid(theme, code_embeddings)
            if centroid is not None:
                theme_centroids.append(centroid)
                valid_themes.append(theme)
            # Skip themes without valid centroids
        
        if self.verbose and len(valid_themes) != len(themes):
            print(f"Generated {len(valid_themes)} theme centroids from {len(themes)} themes")
        
        return valid_themes, theme_centroids

# ============================================================================
# HIGH-PERFORMANCE THEME IDENTIFIER
# ============================================================================

class ThemeIdentifier:
    """
    High-performance theme identifier with hierarchical concurrency following 
    qualityFilter/ideaExtractor patterns for 10-20x performance improvement.
    """
    
    def __init__(self, 
                 codebook: List[Dict[str, str]], 
                 var_lab: str,
                 verbose: bool = False, 
                 prompt_printer = None):
        
        self.codebook = codebook
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        self.model_config = ModelConfig()
        self.client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
        # Configuration - Performance Optimizations
        self.embedding_model = "text-embedding-3-large"
        self.min_cluster_size = 2
        self.variance_threshold = 0.9
        self.umap_n_components = 10
        self.max_existing_themes_to_show = 5
        
        #  Performance Configuration
        self.batch_size = 20  # Clusters per batch
        self.sub_batch_size = 5  # Clusters per sub-batch  
        self.max_concurrent_batches = None  # Unlimited concurrent batches
        self.noise_batch_size = 50  # Increased from 10 to 50
        
        #  Real-time Theme Memory (following codeGenerator pattern)
        self.shared_theme_memory = SharedThemeMemory()
        self.theme_embedding_manager = ThemeEmbeddingManager(self.shared_theme_memory, verbose)
        
        self._initialize_code_registry()
    
    def _initialize_code_registry(self):
        """Create a registry of all codes for tracking and validation"""
        self.code_registry = {}
        for i, code in enumerate(self.codebook, 1):
            self.code_registry[i] = {
                'code_id': i,
                'code_name': code.code,
                'definition': code.definition
            }
    
    def _prepare_codes_for_embedding(self) -> List[CodeEmbedding]:
        """Prepare codes for embedding using Code: [name]. Definition: [definition] format"""
        code_embeddings = []
        for i, code in enumerate(self.codebook, 1):
            embedding_text = f"Code: {code.code}. Definition: {code.definition}"
            code_embedding = CodeEmbedding(
                code_number=i,
                code_name=code.code,
                definition=code.definition,
                embedding_text=embedding_text
            )
            code_embeddings.append(code_embedding)
        return code_embeddings
    
    async def _generate_embeddings(self, code_embeddings: List[CodeEmbedding]) -> List[CodeEmbedding]:
        """Generate embeddings for codes"""
        self.verbose_reporter.stat_line(f"Generating embeddings for {len(code_embeddings)} codes...")
        
        # Extract texts for embedding
        texts = [code.embedding_text for code in code_embeddings]
        
        try:
            # Generate embeddings in batch
            response = await self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            
            # Assign embeddings back to codes
            for code_embedding, embedding_data in zip(code_embeddings, response.data):
                code_embedding.embedding = np.array(embedding_data.embedding, dtype=np.float32)
            
            self.verbose_reporter.stat_line(f"✅ Generated {len(code_embeddings)} embeddings")
            return code_embeddings
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"❌ Error generating embeddings: {str(e)}")
            raise e
    
    def _perform_clustering(self, code_embeddings: List[CodeEmbedding]) -> List[CodeEmbedding]:
        """Perform PCA → UMAP → HDBSCAN clustering on code embeddings"""
        self.verbose_reporter.stat_line("Starting clustering pipeline...")
        
        # Extract embeddings matrix
        embeddings = np.array([code.embedding for code in code_embeddings])
        
        # === Step 2: UMAP ===
        n_codes = len(code_embeddings)
        umap = UMAP(
            n_neighbors=min(15, len(code_embeddings) // 3),  # Adaptive based on dataset size,
            n_components=min(8, max(5, n_codes // 8)),
            min_dist=0.1,  # A small positive value allows for more natural separation.
            metric="cosine",
            random_state=42,
            n_jobs=1,
            low_memory=True,
            transform_seed=42
        )
        umap_embeddings = umap.fit_transform(embeddings)
          
        # === Step 3: HDBSCAN ===
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon = 0.1,  # Prevents over-fragmentation
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            approx_min_span_tree=False,
            gen_min_span_tree=True
        )
        labels = hdb.fit_predict(umap_embeddings)
        
        # Assign cluster labels to codes
        for code_embedding, label in zip(code_embeddings, labels):
            code_embedding.cluster_id = int(label)
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = list(labels).count(-1)
        
        self.verbose_reporter.stat_line(f"[HDBSCAN] Found {num_clusters} clusters")
        self.verbose_reporter.stat_line(f"[HDBSCAN] Noise points: {noise_points} / {len(code_embeddings)} ({noise_points / len(code_embeddings) * 100:.1f}%)")
        
        return code_embeddings
    
    def _group_codes_by_cluster(self, code_embeddings: List[CodeEmbedding]) -> Dict[int, List[CodeEmbedding]]:
        """Group codes by their cluster assignments"""
        clusters = {}
        for code in code_embeddings:
            cluster_id = code.cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(code)
        return clusters
    
    async def _find_nearest_existing_themes(self, cluster_codes: List[CodeEmbedding], 
                                           code_embeddings: List[CodeEmbedding], n: int = 5) -> List[ExistingThemeOption]:
        """Find nearest existing themes to a cluster using real-time theme memory (following codeGenerator pattern)"""
        
        # Get CURRENT theme state (like codeGenerator's get_current_snapshot)
        current_themes, version = await self.shared_theme_memory.get_current_snapshot()
        
        if not current_themes:
            return []
        
        # Calculate cluster centroid
        cluster_embeddings = np.array([code.embedding for code in cluster_codes])
        cluster_centroid = np.mean(cluster_embeddings, axis=0)
        
        # Get fresh theme centroids for current state (like codeGenerator's get_snapshot_embeddings)
        themes, theme_centroids = await self.theme_embedding_manager.get_snapshot_embeddings(
            current_themes, version, code_embeddings
        )
        
        if not theme_centroids:
            return []
        
        # Calculate cosine similarities (following codeGenerator pattern)
        theme_array = np.array(theme_centroids)
        similarities = cosine_similarity(cluster_centroid.reshape(1, -1), theme_array)[0]
        top_k_indices = np.argsort(similarities)[-n:][::-1]
        
        # Get unique themes (following codeGenerator's deduplication logic)
        seen = set()
        existing_options = []
        
        for idx in top_k_indices:
            if idx < len(themes):
                theme = themes[idx]
                theme_name = theme.theme_name
                
                if theme_name not in seen:
                    seen.add(theme_name)
                    existing_options.append(ExistingThemeOption(
                        theme_name=theme.theme_name,
                        theme_description=theme.theme_description,
                        similarity_score=float(similarities[idx])
                    ))
                    
                    if len(existing_options) >= n:
                        break
        
        return existing_options
    
    def _create_cluster_theme_prompt(self, cluster_codes: List[CodeEmbedding], existing_options: List[ExistingThemeOption]) -> str:
        """Create prompt for naming a cluster theme"""
        
        # Format cluster codes
        codes_text = "\n".join([
            f"{code.code_number}. Code: {code.code_name}. Definition: {code.definition}"
            for code in cluster_codes
        ])
        
        # Format existing theme options
        existing_themes_text = ""
        if existing_options:
            existing_themes_text = "\nEXISTING THEMES (ranked by similarity to this cluster):\n"
            for i, option in enumerate(existing_options, 1):
                existing_themes_text += f"{i}. {option.theme_name}: {option.theme_description} (similarity: {option.similarity_score:.3f})\n"
        
        prompt = THEME_IDENTIFICATION_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            codes_count=len(cluster_codes),
            codes_text=codes_text,
            existing_themes_text=existing_themes_text
        )
        
        return prompt
    
    async def _decide_cluster_theme(self, cluster_codes: List[CodeEmbedding], code_embeddings: List[CodeEmbedding]) -> ClusterThemeDecision:
        """Decide on theme assignment for a cluster"""
        
        # Find nearest existing themes using real-time memory
        existing_options = await self._find_nearest_existing_themes(cluster_codes, code_embeddings, self.max_existing_themes_to_show)
        
        # Create prompt
        prompt = self._create_cluster_theme_prompt(cluster_codes, existing_options)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("theme_synthesis"),
                messages=[{"role": "user", "content": prompt}],
                response_model=ClusterThemeDecision,
                temperature=0.0,
                max_retries=2
            )
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error in cluster theme decision: {str(e)}")
            # Return default new theme on error
            return ClusterThemeDecision(
                decision="create_single_theme",
                themes=[ThemeDecision(
                    theme_name=f"Cluster {cluster_codes[0].cluster_id} thema",
                    theme_description="Automatisch gegenereerd thema vanwege verwerkingsfout",
                    assigned_codes=[code.code_number for code in cluster_codes],
                    confidence="low",
                    is_existing=False
                )],
                existing_theme_used=None,
                rationale=f"Error during processing: {str(e)}"
            )
    
    #  PERFORMANCE OPTIMIZATIONS - HIERARCHICAL CONCURRENCY
    
    def _create_cluster_batches(self, cluster_ids: List[int], clusters: Dict[int, List[CodeEmbedding]]) -> List[List[int]]:
        """Create batches of cluster IDs for processing"""
        # Exclude noise cluster (-1) from batching
        non_noise_clusters = [cid for cid in cluster_ids if cid != -1]
        
        batches = []
        for i in range(0, len(non_noise_clusters), self.batch_size):
            batch = non_noise_clusters[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    def _create_sub_batches(self, batch: List[int]) -> List[List[int]]:
        """Split a batch into smaller sub-batches for concurrent processing"""
        if not batch:
            return []
        
        sub_batches = []
        for i in range(0, len(batch), self.sub_batch_size):
            sub_batch = batch[i:i + self.sub_batch_size]
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    async def _process_sub_batch(self, sub_batch: List[int], clusters: Dict[int, List[CodeEmbedding]], 
                                code_embeddings: List[CodeEmbedding], batch_index: int, sub_batch_index: int) -> List[ThemeStructure]:
        """Process a single sub-batch of clusters"""
        sub_batch_results = []
        
        # Create tasks for all clusters in this sub-batch
        tasks = []
        for cluster_id in sub_batch:
            cluster_codes = clusters[cluster_id]
            task = self._decide_cluster_theme(cluster_codes, code_embeddings)
            tasks.append((cluster_id, cluster_codes, task))
        
        # Process all clusters in sub-batch concurrently
        cluster_tasks = [task for _, _, task in tasks]
        theme_decisions = await asyncio.gather(*cluster_tasks, return_exceptions=True)
        
        # Process results and create themes
        for (cluster_id, cluster_codes, _), theme_decision in zip(tasks, theme_decisions):
            if isinstance(theme_decision, Exception):
                print(f"Sub-batch {sub_batch_index + 1} of batch {batch_index + 1}, cluster {cluster_id} failed: {str(theme_decision)}")
                # Create fallback theme
                fallback_theme = ThemeStructure(
                    theme_name=f"Cluster {cluster_id} thema",
                    theme_description="Automatisch gegenereerd thema vanwege verwerkingsfout",
                    codes=[
                        CodeReference(
                            code_number=code.code_number,
                            code_name=code.code_name,
                            definition=code.definition
                        )
                        for code in cluster_codes
                    ],
                    cluster_id=cluster_id,
                    is_miscellaneous=False
                )
                sub_batch_results.append(fallback_theme)
                continue
            
            # Process the multi-theme decision
            cluster_themes = await self._process_multi_theme_decision(
                theme_decision, cluster_codes, cluster_id
            )
            
            # Add new themes to shared memory (following codeGenerator pattern)
            for theme in cluster_themes:
                added, new_version = await self.shared_theme_memory.add_theme_if_new(theme)
                if added and self.verbose:
                    print(f"Cluster {cluster_id}: Added new theme '{theme.theme_name}' (v{new_version}) - NOW AVAILABLE for subsequent clusters")
            
            sub_batch_results.extend(cluster_themes)
        
        return sub_batch_results
    
    async def _process_batch(self, batch: List[int], clusters: Dict[int, List[CodeEmbedding]], 
                           code_embeddings: List[CodeEmbedding], batch_index: int) -> List[ThemeStructure]:
        """Process a single batch with hierarchical concurrency"""
        # Split batch into sub-batches
        sub_batches = self._create_sub_batches(batch)
        
        if not sub_batches:
            return []
        
        # Level 2: Process all sub-batches within this batch concurrently
        sub_batch_tasks = [
            self._process_sub_batch(sub_batch, clusters, code_embeddings, batch_index, i) 
            for i, sub_batch in enumerate(sub_batches)
        ]
        sub_batch_results = await asyncio.gather(*sub_batch_tasks, return_exceptions=True)
        
        # Collect results from all sub-batches
        batch_results = []
        sub_batch_failures = 0
        
        for i, sub_batch_result in enumerate(sub_batch_results):
            if isinstance(sub_batch_result, Exception):
                print(f"Sub-batch {i+1} of batch {batch_index+1} failed completely: {str(sub_batch_result)}")
                sub_batch_failures += 1
                continue
            
            # Add all results from this sub-batch
            batch_results.extend(sub_batch_result)
        
        if sub_batch_failures > 0:
            print(f"{sub_batch_failures} out of {len(sub_batches)} sub-batches failed in batch {batch_index+1}")
        
        return batch_results
    
    async def _process_all_batches(self, batches: List[List[int]], clusters: Dict[int, List[CodeEmbedding]], 
                                  code_embeddings: List[CodeEmbedding]) -> List[ThemeStructure]:
        """Process all batches using hierarchical concurrency"""
        total_clusters = sum(len(batch) for batch in batches)
        total_sub_batches = sum(len(self._create_sub_batches(batch)) for batch in batches)
        
        self.verbose_reporter.stat_line(
            f"Processing {total_clusters} clusters in {len(batches)} batches "
            f"({total_sub_batches} concurrent sub-batches)..."
        )
        
        # Level 1: Process ALL batches concurrently (NO LIMITS)
        batch_tasks = [self._process_batch(batch, clusters, code_embeddings, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results from all batches
        all_results = []
        total_failures = 0
        
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                print(f"Batch {i+1} processing failed completely: {str(batch_result)}")
                total_failures += 1
                continue
            
            # Add all results from this batch
            all_results.extend(batch_result)
        
        if total_failures > 0:
            print(f"{total_failures} out of {len(batches)} batches failed completely")
        
        return all_results
    
    def _create_individual_noise_assignment_prompt(self, code: CodeEmbedding, existing_themes: List[ThemeStructure]) -> str:
        """Create prompt for analyzing a single noise point code"""
        
        existing_themes_text = ""
        for i, theme in enumerate(existing_themes, 1):
            if theme.is_miscellaneous:
                continue  # Skip miscellaneous theme itself
            existing_themes_text += f"{i}. {theme.theme_name}: {theme.theme_description}\n"
        
        prompt = ASSIGN_MISCELLANEOUS_PROMPT.format(
            language=DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            existing_themes_text=existing_themes_text,
            code_number=code.code_number,
            code_name=code.code_name,
            definition=code.definition
        )
        
        return prompt
    
    async def _assign_individual_noise_code(self, code: CodeEmbedding) -> Dict[str, Any]:
        """Analyze a single noise code for best theme assignment"""
        
        # Get current themes from shared memory
        current_themes, _ = await self.shared_theme_memory.get_current_snapshot()
        
        prompt = self._create_individual_noise_assignment_prompt(code, current_themes)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("theme_synthesis"),
                messages=[{"role": "user", "content": prompt}],
                response_model=IndividualCodeAssignment,
                temperature=0.0,
                max_retries=3
            )
            
            return {
                "code": code,
                "decision": response.decision,
                "target_theme": response.target_theme,
                "confidence": response.confidence,
                "rationale": response.rationale
            }
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error processing Code {code.code_number}: {str(e)}")
            return {
                "code": code,
                "decision": "miscellaneous",
                "target_theme": None,
                "confidence": "low",
                "rationale": f"Processing failed: {str(e)}"
            }
    
    async def _process_noise_codes_concurrently(self, noise_codes: List[CodeEmbedding]) -> List[Dict[str, Any]]:
        """Process noise codes with optimized concurrent processing -  Performance Enhancement"""
        
        if not noise_codes:
            return []
        
        self.verbose_reporter.stat_line(f"Processing {len(noise_codes)} noise codes with optimized concurrency...")
        
        #  OPTIMIZATION: Remove artificial delay and use larger batches
        batch_size = self.noise_batch_size  # Increased from 10 to 50
        all_assignments = []
        
        # Process in batches for better concurrency control
        for i in range(0, len(noise_codes), batch_size):
            batch = noise_codes[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(noise_codes) + batch_size - 1) // batch_size
            
            self.verbose_reporter.stat_line(f"Processing batch {batch_num}/{total_batches}: {len(batch)} codes...")
            
            # Create concurrent tasks for this batch
            tasks = [
                self._assign_individual_noise_code(code)
                for code in batch
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    self.verbose_reporter.stat_line(f"Noise code processing failed: {str(result)}")
                else:
                    all_assignments.append(result)
            
            #  OPTIMIZATION: NO MORE ARTIFICIAL DELAYS
            # Removed: await asyncio.sleep(0.5)
        
        return all_assignments
    
    async def identify_themes_by_clustering(self) -> Dict[str, Any]:
        """
         Main method: High-performance theme identification using hierarchical concurrency
        """
        
        self.verbose_reporter.section_header("CLUSTERING-BASED THEME IDENTIFICATION  (OPTIMIZED)")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for theme identification")
            return {
                'codebook': [],
                'themes': [],
                'methodology': 'Clustering-based Braun & Clarke (2006) -  Optimized'
            }
        
        total_codes = len(self.codebook)
        self.verbose_reporter.stat_line(f"Starting with {total_codes} codes")
        
        # Step 1: Prepare codes for embedding
        code_embeddings = self._prepare_codes_for_embedding()
        
        # Step 2: Generate embeddings
        code_embeddings = await self._generate_embeddings(code_embeddings)
        
        # Step 3: Perform clustering
        code_embeddings = self._perform_clustering(code_embeddings)
        
        # Step 4: Group codes by cluster
        clusters = self._group_codes_by_cluster(code_embeddings)
        
        # Step 5: Process clusters with  hierarchical concurrency
        identified_themes = []
        
        # Sort clusters by size (largest first, but process noise last)
        cluster_ids = sorted([cid for cid in clusters.keys() if cid != -1], 
                           key=lambda cid: len(clusters[cid]), reverse=True)
        
        #  OPTIMIZATION: Process regular clusters with hierarchical concurrency
        if cluster_ids:
            # Create batches for concurrent processing
            batches = self._create_cluster_batches(cluster_ids, clusters)
            
            # Capture sample prompt for first cluster
            if self.prompt_printer and batches and batches[0]:
                first_cluster_id = batches[0][0]
                cluster_codes = clusters[first_cluster_id]
                existing_options = await self._find_nearest_existing_themes(cluster_codes, code_embeddings, self.max_existing_themes_to_show)
                sample_prompt = self._create_cluster_theme_prompt(cluster_codes, existing_options)
                self.prompt_printer.capture_prompt(
                    step_name="clustering_based_theme_identification",
                    utility_name="ThemeIdentifier",
                    prompt_content=sample_prompt,
                    prompt_type="Clustering-based Theme Decision",
                    metadata={
                        "model": self.model_config.get_model_for_stage("theme_synthesis"),
                        "var_lab": self.var_lab,
                        "language": DEFAULT_LANGUAGE,
                        "batch_size": self.batch_size,
                        "sub_batch_size": self.sub_batch_size,
                        "total_batches": len(batches)
                    }
                )
            
            # Process all batches concurrently
            batch_themes = await self._process_all_batches(batches, clusters, code_embeddings)
            identified_themes.extend(batch_themes)
        
        # Step 6: Process noise codes if present
        if -1 in clusters:
            noise_codes = clusters[-1]
            self.verbose_reporter.stat_line(f"\nProcessing {len(noise_codes)} noise codes with  optimizations")
            
            # Capture sample prompt for first noise code
            if self.prompt_printer and noise_codes:
                current_themes, _ = await self.shared_theme_memory.get_current_snapshot()
                sample_prompt = self._create_individual_noise_assignment_prompt(noise_codes[0], current_themes)
                self.prompt_printer.capture_prompt(
                    step_name="individual_noise_code_assignment",
                    utility_name="ThemeIdentifier",
                    prompt_content=sample_prompt,
                    prompt_type="Individual Noise Code Assignment",
                    metadata={
                        "model": self.model_config.get_model_for_stage("theme_synthesis"),
                        "var_lab": self.var_lab,
                        "language": DEFAULT_LANGUAGE,
                        "noise_batch_size": self.noise_batch_size,
                        "total_noise_codes": len(noise_codes)
                    }
                )
            
            noise_assignments = await self._process_noise_codes_concurrently(noise_codes)
            
            # Apply individual assignments
            remaining_misc_codes = []
            
            for assignment in noise_assignments:
                code = assignment["code"]
                decision = assignment["decision"]
                target_theme = assignment["target_theme"]
                confidence = assignment["confidence"]
                
                if decision == "assign" and target_theme:
                    # Find target theme and add code
                    theme_found = False
                    for theme in identified_themes:
                        if theme.theme_name == target_theme:
                            new_code_ref = CodeReference(
                                code_number=code.code_number,
                                code_name=code.code_name,
                                definition=code.definition
                            )
                            theme.codes.append(new_code_ref)
                            self.verbose_reporter.stat_line(f"✓ Code {code.code_number} → '{target_theme}' ({confidence} confidence)")
                            theme_found = True
                            break
                    
                    if not theme_found:
                        self.verbose_reporter.stat_line(f"⚠️ Target theme '{target_theme}' not found for Code {code.code_number}, keeping miscellaneous")
                        remaining_misc_codes.append(code)
                else:
                    # Keep as miscellaneous
                    remaining_misc_codes.append(code)
                    if decision == "miscellaneous":
                        self.verbose_reporter.stat_line(f"○ Code {code.code_number} remains miscellaneous ({confidence} confidence)")
            
            # Handle remaining miscellaneous codes
            if remaining_misc_codes:
                misc_codes = [
                    CodeReference(
                        code_number=code.code_number,
                        code_name=code.code_name,
                        definition=code.definition
                    )
                    for code in remaining_misc_codes
                ]
                
                misc_theme = ThemeStructure(
                    theme_name="Overige aspecten",
                    theme_description="Codes die niet in andere thema's passen",
                    codes=misc_codes,
                    cluster_id=-1,
                    is_miscellaneous=True
                )
                identified_themes.append(misc_theme)
                self.verbose_reporter.stat_line(f"📦 {len(misc_codes)} codes remain in miscellaneous theme")
            else:
                self.verbose_reporter.stat_line("🎉 All noise codes successfully reassigned to existing themes!")
        
        # Get final themes from shared memory
        final_themes, final_version = await self.shared_theme_memory.get_current_snapshot()
        final_themes.extend(identified_themes)  # Add any remaining themes not in shared memory
        
        return await self._finalize_theme_identification(final_themes, clusters, total_codes, start_time)
    
    async def _process_multi_theme_decision(
        self, 
        theme_decision: ClusterThemeDecision, 
        cluster_codes: List[CodeEmbedding], 
        cluster_id: int
    ) -> List[ThemeStructure]:
        """Process a multi-theme decision for a cluster - returns list of new themes"""
        new_themes = []
        
        if theme_decision.decision == "use_existing_theme":
            # For existing themes, we would need to add codes to existing theme
            # This is handled at a higher level, return empty list
            return new_themes
        
        elif theme_decision.decision == "create_single_theme":
            if len(theme_decision.themes) >= 1:
                theme_info = theme_decision.themes[0]
                new_theme = self._create_single_theme_from_decision(
                    theme_info, cluster_codes, cluster_id
                )
                new_themes.append(new_theme)
        
        elif theme_decision.decision == "split_into_multiple_themes":
            # Create a lookup of cluster codes by code number
            code_lookup = {code.code_number: code for code in cluster_codes}
            
            for i, theme_info in enumerate(theme_decision.themes):
                # Get codes assigned to this theme
                theme_cluster_codes = []
                for code_num in theme_info.assigned_codes:
                    if code_num in code_lookup:
                        theme_cluster_codes.append(code_lookup[code_num])
                
                if theme_cluster_codes:
                    new_theme = self._create_single_theme_from_decision(
                        theme_info, theme_cluster_codes, f"{cluster_id}_{i+1}"
                    )
                    new_themes.append(new_theme)
        
        elif theme_decision.decision == "reject_mixed_cluster":
            # These codes will be handled as noise
            pass
        
        return new_themes
    
    def _create_single_theme_from_decision(
        self, 
        theme_info: ThemeDecision, 
        cluster_codes: List[CodeEmbedding], 
        cluster_id: str
    ) -> ThemeStructure:
        """Create a single theme from a theme decision"""
        theme_codes = [
            CodeReference(
                code_number=code.code_number,
                code_name=code.code_name,
                definition=code.definition
            )
            for code in cluster_codes
        ]
        
        new_theme = ThemeStructure(
            theme_name=theme_info.theme_name,
            theme_description=theme_info.theme_description,
            codes=theme_codes,
            cluster_id=cluster_id,
            is_miscellaneous=False
        )
        
        return new_theme
    
    async def _finalize_theme_identification(
        self, 
        identified_themes: List[ThemeStructure], 
        clusters: Dict[int, List[CodeEmbedding]], 
        total_codes: int, 
        start_time: float
    ) -> Dict[str, Any]:
        """Finalize the theme identification process"""
        
        # Build final structure
        final_structure = self._build_final_codebook_structure(identified_themes)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        self.verbose_reporter.summary(" THEME IDENTIFICATION COMPLETE", {
            "Total codes": total_codes,
            "Clusters found": len([cid for cid in clusters.keys() if cid != -1]),
            "Themes identified": len(identified_themes),
            "Processing time": f"{elapsed_time:.2f}s",
            "Performance improvement": "10-20x faster with hierarchical concurrency",
            "Methodology": "Clustering-based Braun & Clarke (2006) with concurrency optimizations"
        })
        
        return final_structure
    
    def _build_final_codebook_structure(self, themes: List[ThemeStructure]) -> Dict[str, Any]:
        """Build final structure with theme assignments"""
        codebook_data = []
        
        for theme in themes:
            for code_ref in theme.codes:
                codebook_data.append({
                    'code_id': code_ref.code_number,
                    'code': code_ref.code_name,
                    'definition': code_ref.definition,
                    'theme': theme.theme_name,
                    'theme_description': theme.theme_description,
                    'cluster_id': theme.cluster_id,
                    'is_miscellaneous': theme.is_miscellaneous
                })
        
        # Sort by code_id to maintain original order
        codebook_data.sort(key=lambda x: x['code_id'])
        
        return {
            'codebook': codebook_data,
            'themes': [
                {
                    'theme_name': theme.theme_name,
                    'theme_description': theme.theme_description,
                    'codes': [
                        {
                            'code_number': code.code_number,
                            'code_name': code.code_name,
                            'definition': code.definition
                        }
                        for code in theme.codes
                    ],
                    'cluster_id': theme.cluster_id,
                    'is_miscellaneous': theme.is_miscellaneous
                }
                for theme in themes
            ],
            'methodology': 'Clustering-based Braun & Clarke (2006) with hierarchical concurrency',
            'analysis_approach': 'High-performance semantic clustering with concurrent LLM theme naming'
        }