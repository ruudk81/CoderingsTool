import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import instructor
from openai import AsyncOpenAI

# === CLUSTERING =====================================================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

# === MODELS ========================================================================================================
from pydantic import BaseModel, Field, model_validator

# === CONFIG ========================================================================================================
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
from utils.verboseReporter import VerboseReporter

# === UTILS ========================================================================================================
try:
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

class ClusterThemeDecision(BaseModel):
    """Decision about theme assignment for a cluster"""
    decision: str = Field(description="create_new or use_existing")
    theme_name: str = Field(description="Theme name (new or existing)")
    theme_description: str = Field(description="Theme description")
    existing_theme_used: Optional[str] = Field(description="Name of existing theme if used", default=None)
    confidence: str = Field(description="Confidence level: high, medium, or low")
    rationale: str = Field(description="Explanation of the decision")
    
    @model_validator(mode='after')
    def validate_decision_consistency(self):
        """Ensure decision data is consistent"""
        if self.decision == "use_existing":
            if not self.existing_theme_used:
                raise ValueError("existing_theme_used required when using existing theme")
        return self

class ThemeStructure(BaseModel):
    """Theme with codes following clustering-based methodology"""
    theme_name: str = Field(description="Descriptive theme name in target language")
    theme_description: str = Field(description="Brief explanation of what unites these codes conceptually")
    codes: List[CodeReference] = Field(description="Codes that belong to this theme")
    cluster_id: int = Field(description="Original cluster ID")
    is_miscellaneous: bool = Field(description="Whether this is a miscellaneous theme", default=False)

# ============================================================================
# CLUSTERING-BASED THEME IDENTIFIER V4
# ============================================================================

class ThemeIdentifierV4:
    
    def __init__(self, 
                 codebook: List[Dict[str, str]], 
                 var_lab: str,
                 verbose: bool = False, 
                 prompt_printer = None):
        
        self.codebook = codebook
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose)
        self.prompt_printer = prompt_printer
        self.model_config = ModelConfig()
        self.client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        
        # Configuration
        self.embedding_model = "text-embedding-3-large"
        self.min_cluster_size = 2
        self.variance_threshold = 0.9
        self.umap_n_components = 10
        self.max_existing_themes_to_show = 5
        
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
        
        # === Step 1: PCA ===
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)
        
        pca = PCA()
        pca_embeddings = pca.fit_transform(scaled)
        
        # Find optimal dimensions to retain 90% variance
        total_variance = 0.0
        optimal_dims = 0
        for i, variance in enumerate(pca.explained_variance_ratio_):
            total_variance += variance
            if total_variance >= self.variance_threshold:
                optimal_dims = i + 1
                break
        
        pca_embeddings = pca_embeddings[:, :optimal_dims]
        self.verbose_reporter.stat_line(f"[PCA] Reduced {embeddings.shape[1]} → {optimal_dims} dims ({total_variance * 100:.2f}% variance retained)")
        
        # === Step 2: UMAP ===
        umap = UMAP(
            n_neighbors=5,
            n_components=self.umap_n_components,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            n_jobs=1,
            low_memory=True,
            transform_seed=42
        )
        umap_embeddings = umap.fit_transform(pca_embeddings)
        self.verbose_reporter.stat_line(f"[UMAP] Reduced {optimal_dims} → {self.umap_n_components} dims")
        
        # === Step 3: HDBSCAN ===
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=None,
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
    
    def _find_nearest_existing_themes(self, cluster_codes: List[CodeEmbedding], existing_themes: List[ThemeStructure], n: int = 5) -> List[ExistingThemeOption]:
        """Find nearest existing themes to a cluster"""
        if not existing_themes:
            return []
        
        # Calculate cluster centroid
        cluster_embeddings = np.array([code.embedding for code in cluster_codes])
        cluster_centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        
        # Calculate similarities to existing themes (use first code embedding as theme representation)
        existing_options = []
        for theme in existing_themes:
            if theme.codes:
                # Use first code in theme as theme representative (simple approach)
                theme_embedding = None
                for code in cluster_codes:  # Find embedding for first code in theme
                    if code.code_number == theme.codes[0].code_number:
                        theme_embedding = code.embedding.reshape(1, -1)
                        break
                
                if theme_embedding is not None:
                    similarity = cosine_similarity(cluster_centroid, theme_embedding)[0][0]
                    existing_options.append(ExistingThemeOption(
                        theme_name=theme.theme_name,
                        theme_description=theme.theme_description,
                        similarity_score=float(similarity)
                    ))
        
        # Sort by similarity and return top n
        existing_options.sort(key=lambda x: x.similarity_score, reverse=True)
        return existing_options[:n]
    
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
        
        prompt = f"""You are a qualitative researcher specializing in thematic analysis following Braun & Clarke (2006) methodology.
You have been given a cluster of semantically related codes that were grouped using HDBSCAN clustering based on their embedding similarity.

SURVEY QUESTION:
{self.var_lab}

CLUSTER CODES ({len(cluster_codes)} codes):
{codes_text}
{existing_themes_text}

YOUR TASK:
Analyze this cluster of codes and decide whether to:
1. CREATE A NEW THEME that captures the shared conceptual pattern of these codes, OR
2. USE AN EXISTING THEME if one of the existing themes above adequately represents this cluster

DECISION CRITERIA:
- For CREATE NEW: The codes share a coherent conceptual pattern that is distinct from existing themes
- For USE EXISTING: An existing theme adequately captures the essence of these codes (similarity > 0.7 is a good indicator)
- Focus on conceptual meaning rather than surface-level keyword matching
- Consider the survey question context

BRAUN & CLARKE PRINCIPLES:
- Themes should capture coherent patterns of meaning
- Focus on conceptual importance, not just frequency
- Each theme should tell a meaningful story about the data
- Themes should be distinct yet coherent

OUTPUT FORMAT (JSON):
{{
  "decision": "create_new|use_existing",
  "theme_name": "[Theme name in {DEFAULT_LANGUAGE}]",
  "theme_description": "[Brief description of what unites these codes conceptually]",
  "existing_theme_used": "[Name of existing theme if used, or null]",
  "confidence": "high|medium|low",
  "rationale": "[Detailed explanation of your decision]"
}}

IMPORTANT:
- Theme names and descriptions must be in {DEFAULT_LANGUAGE}
- If using existing theme, use the EXACT theme name from the list above
- Base your decision on conceptual fit, not just similarity scores
- Be conservative: prefer existing themes when they adequately fit

Return ONLY the JSON object."""
        
        return prompt
    
    async def _decide_cluster_theme(self, cluster_codes: List[CodeEmbedding], existing_themes: List[ThemeStructure]) -> ClusterThemeDecision:
        """Decide on theme assignment for a cluster"""
        
        # Find nearest existing themes
        existing_options = self._find_nearest_existing_themes(cluster_codes, existing_themes, self.max_existing_themes_to_show)
        
        # Create prompt
        prompt = self._create_cluster_theme_prompt(cluster_codes, existing_options)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("theme_synthesis"),
                messages=[{"role": "user", "content": prompt}],
                response_model=ClusterThemeDecision,
                temperature=0.1,
                max_retries=2
            )
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error in cluster theme decision: {str(e)}")
            # Return default new theme on error
            return ClusterThemeDecision(
                decision="create_new",
                theme_name=f"Cluster {cluster_codes[0].cluster_id} thema",
                theme_description="Automatisch gegenereerd thema vanwege verwerkingsfout",
                confidence="low",
                rationale=f"Error during processing: {str(e)}"
            )
    
    async def identify_themes_by_clustering(self) -> Dict[str, Any]:
        """
        Main method: Identify themes using clustering-based approach
        """
        
        self.verbose_reporter.section_header("CLUSTERING-BASED THEME IDENTIFICATION V4")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for theme identification")
            return {
                'codebook': [],
                'themes': [],
                'methodology': 'Clustering-based Braun & Clarke (2006)'
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
        
        # Step 5: Process each cluster to determine themes
        identified_themes = []
        
        # Sort clusters by size (largest first, but process noise last)
        cluster_ids = sorted([cid for cid in clusters.keys() if cid != -1], 
                           key=lambda cid: len(clusters[cid]), reverse=True)
        if -1 in clusters:
            cluster_ids.append(-1)  # Process noise last
        
        # Capture sample prompt for first cluster
        first_cluster = True
        
        for cluster_id in cluster_ids:
            cluster_codes = clusters[cluster_id]
            
            if cluster_id == -1:
                # Handle noise as miscellaneous
                misc_codes = [
                    CodeReference(
                        code_number=code.code_number,
                        code_name=code.code_name,
                        definition=code.definition
                    )
                    for code in cluster_codes
                ]
                
                misc_theme = ThemeStructure(
                    theme_name="Overige aspecten",
                    theme_description="Codes die niet in andere thema's passen",
                    codes=misc_codes,
                    cluster_id=cluster_id,
                    is_miscellaneous=True
                )
                identified_themes.append(misc_theme)
                
                self.verbose_reporter.stat_line(f"📦 Cluster {cluster_id} (noise) → Miscellaneous theme with {len(misc_codes)} codes")
                continue
            
            self.verbose_reporter.stat_line(f"\nProcessing Cluster {cluster_id}: {len(cluster_codes)} codes")
            
            # Decide on theme for this cluster
            theme_decision = await self._decide_cluster_theme(cluster_codes, identified_themes)
            
            # Capture prompt for first cluster
            if first_cluster and self.prompt_printer:
                existing_options = self._find_nearest_existing_themes(cluster_codes, identified_themes, self.max_existing_themes_to_show)
                sample_prompt = self._create_cluster_theme_prompt(cluster_codes, existing_options)
                self.prompt_printer.capture_prompt(
                    step_name="clustering_based_theme_identification",
                    utility_name="ThemeIdentifierV4",
                    prompt_content=sample_prompt,
                    prompt_type="Clustering-based Theme Decision"
                )
                first_cluster = False
            
            if theme_decision.decision == "use_existing":
                # Find existing theme and add codes to it
                target_theme = None
                for theme in identified_themes:
                    if theme.theme_name == theme_decision.existing_theme_used:
                        target_theme = theme
                        break
                
                if target_theme:
                    # Add codes to existing theme
                    new_codes = [
                        CodeReference(
                            code_number=code.code_number,
                            code_name=code.code_name,
                            definition=code.definition
                        )
                        for code in cluster_codes
                    ]
                    target_theme.codes.extend(new_codes)
                    
                    self.verbose_reporter.stat_line(
                        f"✅ Cluster {cluster_id} → Existing theme '{theme_decision.existing_theme_used}' ({theme_decision.confidence} confidence)"
                    )
                else:
                    # Fallback: create new theme if existing not found
                    self.verbose_reporter.stat_line(f"⚠️ Existing theme '{theme_decision.existing_theme_used}' not found, creating new theme")
                    theme_decision.decision = "create_new"
            
            if theme_decision.decision == "create_new":
                # Create new theme
                theme_codes = [
                    CodeReference(
                        code_number=code.code_number,
                        code_name=code.code_name,
                        definition=code.definition
                    )
                    for code in cluster_codes
                ]
                
                new_theme = ThemeStructure(
                    theme_name=theme_decision.theme_name,
                    theme_description=theme_decision.theme_description,
                    codes=theme_codes,
                    cluster_id=cluster_id,
                    is_miscellaneous=False
                )
                identified_themes.append(new_theme)
                
                self.verbose_reporter.stat_line(
                    f"✅ Cluster {cluster_id} → New theme '{theme_decision.theme_name}' ({theme_decision.confidence} confidence)"
                )
        
        # Build final structure
        final_structure = self._build_final_codebook_structure(identified_themes)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        self.verbose_reporter.summary("CLUSTERING-BASED THEME IDENTIFICATION COMPLETE", {
            "Total codes": total_codes,
            "Clusters found": len([cid for cid in clusters.keys() if cid != -1]),
            "Themes identified": len(identified_themes),
            "Methodology": "Clustering-based Braun & Clarke (2006)",
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print detailed results if verbose
        if self.verbose and identified_themes:
            print("\nIdentified Themes (by cluster):")
            for theme in identified_themes:
                cluster_info = f"Cluster {theme.cluster_id}" if not theme.is_miscellaneous else "Noise"
                print(f"\n  {cluster_info} - {theme.theme_name}")
                print(f"    Description: {theme.theme_description}")
                print(f"    Codes ({len(theme.codes)}): {', '.join([f'{c.code_number}. {c.code_name[:30]}...' if len(c.code_name) > 30 else f'{c.code_number}. {c.code_name}' for c in theme.codes[:5]])}")
                if len(theme.codes) > 5:
                    print(f"    ... and {len(theme.codes) - 5} more codes")
        
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
            'methodology': 'Clustering-based Braun & Clarke (2006) - PCA → UMAP → HDBSCAN',
            'analysis_approach': 'Semantic clustering with LLM theme naming'
        }