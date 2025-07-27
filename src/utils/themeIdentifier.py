import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
from typing import List, Dict, Any
import instructor
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
from utils.verboseReporter import VerboseReporter
from prompts import THEME_IDENTIFICATION_PROMPT, DOMAIN_CLUSTERING_PROMPT, THEME_SYNTHESIS_PROMPT

# === UTILS ========================================================================================================
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

class ThemeIdentifier:
    
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
        
        # MapReduce configuration
        self.batch_size = 10
        self.code_registry = {}
        self._initialize_code_registry()
        
    def _format_codes_for_prompt(self) -> str:
      
        if not self.codebook:
            return "No codes available"
        
        formatted_codes = []
        for i, code in enumerate(self.codebook): 
            code_text = code.code or f"Code {i+1}"
            definition = code.definition or "No definition available"
            formatted_codes.append(f"{i+1}. {code_text}: {definition}")
            
        return "\n".join(formatted_codes)
    
    async def _identify_themes_async(self) -> models.ThemeAnalysis:
      
        # Format codes for prompt
        codes_text = self._format_codes_for_prompt()
        
        # Build prompt using Braun & Clarke methodology
        prompt = THEME_IDENTIFICATION_PROMPT.format(
            language = DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            codes=codes_text
        )
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="theme_identification",
                utility_name="ThemeIdentifier",
                prompt_content=prompt,
                prompt_type="Theme Identification"
            )
        
        try:
            # Get structured response using instructor
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("hierarchical_organisation"),   
                messages=[{"role": "user", "content": prompt}],
                response_model=models.ThemeAnalysis,
                temperature=0.3,
                max_retries=3
            )
            
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error identifying themes: {str(e)}")
            
            # Return empty analysis on error
            return models.ThemeAnalysis(
                initial_observations=["Error occurred during theme identification"],
                suggested_themes=[],
                reflection={
                    "broad_or_narrow_themes": "Analysis failed due to error",
                    "contradictions_or_unexpected_patterns": "Could not analyze",
                    "potential_subthemes": "Analysis incomplete", 
                    "unclassified_codes": str([code.code for code in self.codebook])
                }
            )
    
    def identify_themes(self) -> Dict[str, Any]:
       
        self.verbose_reporter.section_header("THEME IDENTIFICATION")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for theme identification")
            return {
                'suggested_themes': [],
                'theme_analysis': models.ThemeAnalysis(
                    initial_observations=["No codes provided for analysis"],
                    suggested_themes=[],
                    reflection={
                        "broad_or_narrow_themes": "No analysis possible - no codes",
                        "contradictions_or_unexpected_patterns": "N/A",
                        "potential_subthemes": "N/A",
                        "unclassified_codes": "N/A"
                    }
                )
            }
        
        self.verbose_reporter.stat_line(f"Analyzing {len(self.codebook)} codes for theme patterns")
        
        # Run async theme identification
        theme_analysis = asyncio.run(self._identify_themes_async())
        
        elapsed_time = time.time() - start_time
        
        # Report results
        num_themes = len(theme_analysis.suggested_themes)
        self.verbose_reporter.summary("THEME IDENTIFICATION COMPLETE", {
            "Input codes": len(self.codebook),
            "Themes identified": num_themes,
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print themes if verbose
        if self.verbose and num_themes > 0:
            print("\nIdentified themes:")
            for i, theme in enumerate(theme_analysis.suggested_themes):
                print(f"  {i+1}. {theme.theme_name}")
                print(f"     Concept: {theme.concept}")
                print(f"     Codes ({len(theme.codes)}): {', '.join(theme.codes)}")
                
        # Check for unclassified codes
        all_theme_codes = set()
        for theme in theme_analysis.suggested_themes:
            all_theme_codes.update(theme.codes)
            
        codebook_codes = {code.code for code in self.codebook}
        unclassified = codebook_codes - all_theme_codes
        
        if unclassified and self.verbose:
            print(f"\nUnclassified codes ({len(unclassified)}): {', '.join(unclassified)}")
            
        return {
            'suggested_themes': theme_analysis.suggested_themes,
            'theme_analysis': theme_analysis,
            'stats': {
                'total_codes': len(self.codebook),
                'themes_identified': num_themes,
                'codes_in_themes': len(all_theme_codes),
                'unclassified_codes': len(unclassified)
            }
        }
    
    # === MAPREDUCE HIERARCHICAL THEME IDENTIFICATION ========================================================================================================
    
    def _initialize_code_registry(self):
        """Create a master registry of all codes for tracking"""
        for i, code in enumerate(self.codebook, 1):
            self.code_registry[i] = {
                'code_id': i,
                'code_text': code.code,
                'definition': code.definition,
                'domain_assignment': None,
                'theme_assignment': None
            }
    
    def _create_code_batches(self) -> List[List[Dict]]:
        """Divide codes into manageable batches for Stage 1"""
        batches = []
        for i in range(0, len(self.codebook), self.batch_size):
            batch = []
            for j, code in enumerate(self.codebook[i:i+self.batch_size], start=i+1):
                batch.append({
                    'number': j,
                    'code': code.code,
                    'definition': code.definition
                })
            batches.append(batch)
        return batches
    
    def _format_batch_for_prompt(self, batch: List[Dict]) -> str:
        """Format a batch of codes for the domain clustering prompt"""
        formatted_codes = []
        for code_info in batch:
            formatted_codes.append(f"{code_info['number']}. {code_info['code']}: {code_info['definition']}")
        return "\n".join(formatted_codes)
    
    async def _identify_domains_for_batch(self, batch: List[Dict], batch_num: int, total_batches: int) -> models.DomainClusteringResult:
        """Stage 1: Identify domains for a batch of codes"""
        codes_text = self._format_batch_for_prompt(batch)
        
        prompt = DOMAIN_CLUSTERING_PROMPT.format(
            system_message=f"Act as a {DEFAULT_LANGUAGE} qualitative data analyst specializing in thematic analysis.",
            batch_number=batch_num,
            total_batches=total_batches,
            survey_question=self.var_lab,
            codes_batch=codes_text,
            language=DEFAULT_LANGUAGE
        )
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="domain_clustering",
                utility_name="ThemeIdentifier",
                prompt_content=prompt,
                prompt_type=f"Domain Clustering - Batch {batch_num}"
            )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("domain_clustering"),
                messages=[{"role": "user", "content": prompt}],
                response_model=models.DomainClusteringResult,
                temperature=0.3,
                max_retries=3
            )
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error in domain clustering for batch {batch_num}: {str(e)}")
            # Return empty result on error
            return models.DomainClusteringResult(
                batch_id=batch_num,
                identified_domains=[],
                processing_notes=f"Error occurred: {str(e)}"
            )
    
    def _consolidate_domains(self, domain_results: List[models.DomainClusteringResult]) -> List[models.DomainDefinition]:
        """Consolidate all domains from Stage 1 batches"""
        all_domains = []
        for result in domain_results:
            all_domains.extend(result.identified_domains)
        return all_domains
    
    def _format_domains_for_synthesis(self, domains: List[models.DomainDefinition]) -> str:
        """Format domains for the theme synthesis prompt"""
        formatted_domains = []
        for i, domain in enumerate(domains, 1):
            codes_text = ", ".join([f"{code.code_number}: {code.code_name}" for code in domain.codes])
            formatted_domains.append(
                f"Domain {i}: {domain.domain_name}\n"
                f"Description: {domain.domain_description}\n"
                f"Codes: {codes_text}\n"
            )
        return "\n".join(formatted_domains)
    
    async def _synthesize_themes(self, all_domains: List[models.DomainDefinition]) -> models.HierarchicalStructure:
        """Stage 2: Organize domains into themes"""
        domains_text = self._format_domains_for_synthesis(all_domains)
        total_codes = len(self.codebook)
        
        prompt = THEME_SYNTHESIS_PROMPT.format(
            system_message=f"Act as a {DEFAULT_LANGUAGE} qualitative data analyst specializing in thematic analysis.",
            survey_question=self.var_lab,
            all_domains=domains_text,
            total_codes=total_codes,
            language=DEFAULT_LANGUAGE
        )
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="theme_synthesis",
                utility_name="ThemeIdentifier",
                prompt_content=prompt,
                prompt_type="Theme Synthesis"
            )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("theme_synthesis"),
                messages=[{"role": "user", "content": prompt}],
                response_model=models.HierarchicalStructure,
                temperature=0.2,  # Lower temperature for consistency
                max_retries=3
            )
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error in theme synthesis: {str(e)}")
            # Return empty structure on error
            return models.HierarchicalStructure(
                themes=[],
                coverage_statistics=models.CoverageStatistics(
                    total_codes=total_codes,
                    classified_codes=0,
                    coverage_percentage=0.0,
                    themes_count=0,
                    domains_count=0,
                    avg_codes_per_domain=0.0
                ),
                quality_notes=f"Error occurred during synthesis: {str(e)}"
            )
    
    def _build_final_codebook_structure(self, hierarchical_result: models.HierarchicalStructure) -> Dict[str, Any]:
        """Build final structure with complete traceability"""
        
        # Update code registry with assignments
        for theme in hierarchical_result.themes:
            for domain in theme.domains:
                for code_assignment in domain.codes:
                    code_id = code_assignment.code_number
                    if code_id in self.code_registry:
                        self.code_registry[code_id]['domain_assignment'] = domain.domain_name
                        self.code_registry[code_id]['theme_assignment'] = theme.theme_name
        
        # Build the final codebook structure
        codebook_data = []
        for code_id, code_record in self.code_registry.items():
            codebook_data.append({
                'code_id': code_id,
                'code': code_record['code_text'],
                'definition': code_record['definition'],
                'domain': code_record['domain_assignment'],
                'theme': code_record['theme_assignment']
            })
        
        return {
            'codebook': codebook_data,
            'hierarchy': hierarchical_result,
            'coverage_lookup': hierarchical_result.get_code_lookup()
        }
    
    async def identify_themes_hierarchical(self) -> Dict[str, Any]:
        """Main method: Two-stage hierarchical identification using MapReduce approach"""
        
        self.verbose_reporter.section_header("HIERARCHICAL THEME IDENTIFICATION")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for hierarchical theme identification")
            return {
                'codebook': [],
                'hierarchy': None,
                'coverage_lookup': {}
            }
        
        total_codes = len(self.codebook)
        self.verbose_reporter.stat_line(f"Processing {total_codes} codes with MapReduce approach (batch size: {self.batch_size})")
        
        # Stage 1: Parallel domain clustering
        batches = self._create_code_batches()
        self.verbose_reporter.stat_line(f"Created {len(batches)} batches for parallel processing")
        
        domain_tasks = []
        for i, batch in enumerate(batches, 1):
            task = self._identify_domains_for_batch(batch, i, len(batches))
            domain_tasks.append(task)
        
        # Execute all domain clustering tasks in parallel
        self.verbose_reporter.stat_line("Starting parallel domain clustering...")
        domain_results = await asyncio.gather(*domain_tasks)
        
        # Consolidate all domains
        all_domains = self._consolidate_domains(domain_results)
        self.verbose_reporter.stat_line(f"Consolidated {len(all_domains)} domains from all batches")
        
        # Stage 2: Theme synthesis
        self.verbose_reporter.stat_line("Starting theme synthesis...")
        hierarchical_structure = await self._synthesize_themes(all_domains)
        
        # Build final structure with traceability
        final_structure = self._build_final_codebook_structure(hierarchical_structure)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        coverage = hierarchical_structure.coverage_statistics
        self.verbose_reporter.summary("HIERARCHICAL THEME IDENTIFICATION COMPLETE", {
            "Total codes": coverage.total_codes,
            "Classified codes": coverage.classified_codes,
            "Coverage": f"{coverage.coverage_percentage:.1f}%",
            "Themes": coverage.themes_count,
            "Domains": coverage.domains_count,
            "Avg codes/domain": f"{coverage.avg_codes_per_domain:.1f}",
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print detailed results if verbose
        if self.verbose and hierarchical_structure.themes:
            print("\nHierarchical Structure:")
            for i, theme in enumerate(hierarchical_structure.themes):
                print(f"\n  Theme {i+1}: {theme.theme_name}")
                print(f"    Concept: {theme.theme_concept}")
                for j, domain in enumerate(theme.domains):
                    print(f"    Domain {j+1}: {domain.domain_name} ({len(domain.codes)} codes)")
                    code_names = [code.code_name[:50] + "..." if len(code.code_name) > 50 else code.code_name for code in domain.codes]
                    print(f"      Codes: {', '.join(code_names)}")
        
        return final_structure
    
    # === VALIDATION FUNCTIONS ========================================================================================================
    
    def validate_hierarchy_completeness(self, hierarchy: models.HierarchicalStructure) -> Dict[str, Any]:
        """Validate that no codes are lost in the hierarchy"""
        
        # Extract all code IDs from hierarchy
        hierarchical_code_ids = set()
        for theme in hierarchy.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    hierarchical_code_ids.add(code.code_number)
        
        # Compare with original
        original_code_ids = set(range(1, len(self.codebook) + 1))
        
        missing_codes = original_code_ids - hierarchical_code_ids
        duplicate_codes = []
        
        # Check for duplicates
        all_codes = []
        for theme in hierarchy.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    all_codes.append(code.code_number)
        
        for code_id in set(all_codes):
            if all_codes.count(code_id) > 1:
                duplicate_codes.append(code_id)
        
        return {
            'complete': len(missing_codes) == 0 and len(duplicate_codes) == 0,
            'missing_codes': list(missing_codes),
            'duplicate_codes': duplicate_codes,
            'coverage_percentage': (len(hierarchical_code_ids) / len(original_code_ids)) * 100 if original_code_ids else 0,
            'total_codes_processed': len(hierarchical_code_ids),
            'total_codes_original': len(original_code_ids)
        }
    
    def get_coverage_report(self, hierarchy: models.HierarchicalStructure) -> str:
        """Generate a human-readable coverage report"""
        validation = self.validate_hierarchy_completeness(hierarchy)
        
        report = f"Coverage Report:\n"
        report += f"- Total codes: {validation['total_codes_original']}\n"
        report += f"- Processed codes: {validation['total_codes_processed']}\n"
        report += f"- Coverage: {validation['coverage_percentage']:.1f}%\n"
        
        if validation['missing_codes']:
            report += f"- Missing codes: {validation['missing_codes']}\n"
        
        if validation['duplicate_codes']:
            report += f"- Duplicate codes: {validation['duplicate_codes']}\n"
        
        report += f"- Status: {'✓ Complete' if validation['complete'] else '✗ Incomplete'}\n"
        
        return report