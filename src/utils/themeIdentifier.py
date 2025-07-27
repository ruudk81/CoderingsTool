import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
from typing import List, Dict, Any
import instructor
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
from pydantic import BaseModel, Field, model_validator

# === CONFIG ========================================================================================================
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
from utils.verboseReporter import VerboseReporter
from prompts import HIERARCHY_MAP_PROMPT, HIERARCHY_REDUCE_PROMPT

# === UTILS ========================================================================================================
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class CodeAssignment(BaseModel):
    """Assignment of a code to a domain with rationale"""
    code_number: int = Field(description="The original code number (1-64)")
    code_name: str = Field(description="The original code name")
    fit_rationale: str = Field(description="Brief explanation why this code fits this domain")

class DomainDefinition(BaseModel):
    """Domain level grouping of related codes"""
    domain_name: str = Field(description="Clear, descriptive name for the domain")
    domain_description: str = Field(description="Brief description of what unites these codes")
    codes: List[CodeAssignment] = Field(description="Codes belonging to this domain")
    
class ThemeDefinition(BaseModel):
    """High-level theme containing multiple domains"""
    theme_name: str = Field(description="Conceptual name for the theme")
    theme_concept: str = Field(description="Explanation of the overarching concept")
    domains: List[DomainDefinition] = Field(description="Domains belonging to this theme")
    
class CodeInHierarchy(BaseModel):
    """Simplified code reference in hierarchy"""
    code_number: int = Field(description="Original code number")
    code_name: str = Field(description="Original code name")

class DomainInHierarchy(BaseModel):
    """Domain in the simplified hierarchy"""
    domain_name: str = Field(description="Domain name")
    codes: List[CodeInHierarchy] = Field(description="Codes in this domain")

class ThemeInHierarchy(BaseModel):
    """Theme in the simplified hierarchy"""
    theme_name: str = Field(description="Theme name")
    domains: List[DomainInHierarchy] = Field(description="Domains in this theme")

class BatchHierarchy(BaseModel):
    """Output for Map stage: Complete hierarchy per batch"""
    batch_id: int = Field(description="Identifier for this batch")
    themes: List[ThemeInHierarchy] = Field(description="Complete hierarchy for this batch")
    
    @model_validator(mode='after')
    def validate_all_codes_present(self):
        """Ensure all codes from the batch are present in the hierarchy"""
        # Extract all code numbers from the hierarchy
        found_codes = set()
        for theme in self.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    found_codes.add(code.code_number)
        
        # Store for later checking
        self._found_codes = found_codes
        return self

class ThemeTransformation(BaseModel):
    """Track how themes were transformed during consolidation"""
    original_themes: List[str] = Field(description="Original theme names from batches")
    final_theme: str = Field(description="Final consolidated theme name")
    transformation_type: str = Field(description="Type: merged, renamed, unchanged")

class DomainTransformation(BaseModel):
    """Track how domains were transformed during consolidation"""
    original_domains: List[str] = Field(description="Original domain names from batches")
    final_domain: str = Field(description="Final consolidated domain name")
    transformation_type: str = Field(description="Type: merged, renamed, unchanged")
    theme: str = Field(description="Theme this domain belongs to")

class ConsolidatedHierarchy(BaseModel):
    """Enhanced hierarchical structure with transformation tracking"""
    themes: List[ThemeDefinition] = Field(description="All themes with their domains and codes")
    theme_transformations: List[ThemeTransformation] = Field(description="How themes were consolidated")
    domain_transformations: List[DomainTransformation] = Field(description="How domains were consolidated")
    
    @model_validator(mode='after')
    def validate_all_codes_preserved(self):
        """Ensure no codes were lost during consolidation"""
        found_codes = set()
        for theme in self.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    found_codes.add(code.code_number)
        self._found_codes = found_codes
        return self

class HierarchicalStructure(BaseModel):
    """Complete three-level hierarchical structure"""
    themes: List[ThemeDefinition] = Field(description="All themes with their domains and codes")
    
    def get_code_lookup(self) -> Dict[int, Dict[str, str]]:
        """Build lookup table: code_id -> {domain, theme}"""
        lookup = {}
        for theme in self.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    lookup[code.code_number] = {
                        'domain': domain.domain_name,
                        'theme': theme.theme_name
                    }
        return lookup

# ============================================================================
# The theme identifier
# ============================================================================

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
        self.batch_size = 10  # Reduced from 15 to 10 for better accuracy
        self.code_registry = {}
        self._initialize_code_registry()
        self.max_hierarchy_retries = 5  # Increased retry attempts
    
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
    
    def _add_missing_codes_to_batch(self, hierarchy: BatchHierarchy, batch: List[Dict], missing_code_numbers: set) -> BatchHierarchy:
        """Add missing codes to a batch hierarchy under 'Overige' theme"""
        # Find or create 'Overige' theme
        overige_theme = None
        for theme in hierarchy.themes:
            if theme.theme_name.lower() in ["overige", "miscellaneous", "diversen"]:
                overige_theme = theme
                break
        
        if not overige_theme:
            # Create new Overige theme
            overige_theme = ThemeInHierarchy(
                theme_name="Overige",
                domains=[]
            )
            hierarchy.themes.append(overige_theme)
        
        # Find or create 'Overige aspecten' domain
        overige_domain = None
        for domain in overige_theme.domains:
            if domain.domain_name.lower() in ["overige aspecten", "miscellaneous aspects", "diverse aspecten"]:
                overige_domain = domain
                break
        
        if not overige_domain:
            overige_domain = DomainInHierarchy(
                domain_name="Overige aspecten",
                codes=[]
            )
            overige_theme.domains.append(overige_domain)
        
        # Add missing codes
        code_lookup = {code['number']: code for code in batch}
        for code_num in sorted(missing_code_numbers):
            if code_num in code_lookup:
                code_info = code_lookup[code_num]
                overige_domain.codes.append(
                    CodeInHierarchy(
                        code_number=code_num,
                        code_name=code_info['code']
                    )
                )
        
        return hierarchy
    
    def _create_fallback_hierarchy(self, batch: List[Dict], batch_num: int) -> BatchHierarchy:
        """Create a fallback hierarchy with all codes in 'Overige' theme"""
        self.verbose_reporter.stat_line(f"🆘 Creating fallback hierarchy for batch {batch_num}")
        
        # Put all codes in a single Overige theme
        codes = []
        for code_info in batch:
            codes.append(
                CodeInHierarchy(
                    code_number=code_info['number'],
                    code_name=code_info['code']
                )
            )
        
        return BatchHierarchy(
            batch_id=batch_num,
            themes=[
                ThemeInHierarchy(
                    theme_name="Overige",
                    domains=[
                        DomainInHierarchy(
                            domain_name="Alle codes",
                            codes=codes
                        )
                    ]
                )
            ]
        )
    
    async def _create_hierarchy_for_batch(self, batch: List[Dict], batch_num: int, total_batches: int) -> BatchHierarchy:
        """Map stage: Create complete hierarchy for a batch of codes with validation and retry"""
        codes_text = self._format_batch_for_prompt(batch)
        expected_code_numbers = [code['number'] for code in batch]
        
        prompt = HIERARCHY_MAP_PROMPT.format(
            batch_number=batch_num,
            survey_question=self.var_lab,
            codes_batch=codes_text,
            codes_to_include=', '.join(str(n) for n in expected_code_numbers),
            language=DEFAULT_LANGUAGE
        )
        
        # Capture prompt if printer is available (only print first batch as sample)
        if self.prompt_printer and batch_num == 1:
            self.prompt_printer.capture_prompt(
                step_name="hierarchy_map",
                utility_name="ThemeIdentifier",
                prompt_content=prompt,
                prompt_type="Hierarchy Creation (Sample Batch 1)"
            )
        
        # Try multiple attempts with validation
        for attempt in range(self.max_hierarchy_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_config.get_model_for_stage("domain_clustering"),
                    messages=[{"role": "user", "content": prompt}],
                    response_model=BatchHierarchy,
                    temperature=0.1 if attempt > 0 else 0.3,  # Lower temperature on retries
                    max_retries=3
                )
                
                # Validate that all expected codes are present
                found_codes = getattr(response, '_found_codes', set())
                if not found_codes:
                    # Manually extract codes if validator didn't run
                    found_codes = set()
                    for theme in response.themes:
                        for domain in theme.domains:
                            for code in domain.codes:
                                found_codes.add(code.code_number)
                
                missing_codes = set(expected_code_numbers) - found_codes
                
                if not missing_codes:
                    # All codes present, return success
                    if attempt > 0:
                        self.verbose_reporter.stat_line(f"✅ Batch {batch_num}: All {len(expected_code_numbers)} codes included on attempt {attempt + 1}")
                    return response
                else:
                    self.verbose_reporter.stat_line(
                        f"⚠️  Batch {batch_num} attempt {attempt + 1}: Missing {len(missing_codes)} codes: {sorted(missing_codes)}"
                    )
                    
                    # If this is the last attempt, add missing codes programmatically
                    if attempt == self.max_hierarchy_retries - 1:
                        response = self._add_missing_codes_to_batch(response, batch, missing_codes)
                        self.verbose_reporter.stat_line(f"🔧 Batch {batch_num}: Added {len(missing_codes)} missing codes to 'Overige' theme")
                        return response
                        
            except Exception as e:
                self.verbose_reporter.stat_line(f"Error in hierarchy creation for batch {batch_num}, attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_hierarchy_retries - 1:
                    # Last attempt failed, create hierarchy with all codes in miscellaneous
                    return self._create_fallback_hierarchy(batch, batch_num)
        
        # Should not reach here, but just in case
        return self._create_fallback_hierarchy(batch, batch_num)
    
    def _format_hierarchies_for_reduction(self, batch_hierarchies: List[BatchHierarchy]) -> str:
        """Format batch hierarchies as readable codebooks for reduce prompt"""
        formatted_parts = []
        
        for hierarchy in batch_hierarchies:
            # Count codes for verification
            total_codes = sum(len(domain.codes) for theme in hierarchy.themes for domain in theme.domains)
            total_domains = sum(len(theme.domains) for theme in hierarchy.themes)
            
            codebook_text = f"CODEBOOK {hierarchy.batch_id}\n"
            codebook_text += "=" * 60 + "\n\n"
            
            for theme in hierarchy.themes:
                codebook_text += f"THEME: {theme.theme_name}\n\n"
                
                for domain in theme.domains:
                    codebook_text += f"  DOMAIN: {domain.domain_name}\n"
                    
                    for code in domain.codes:
                        codebook_text += f"    Code {code.code_number}: {code.code_name}\n"
                    
                    codebook_text += "\n"  # Space between domains
            
            # Add verification summary
            codebook_text += "-" * 60 + "\n"
            codebook_text += f"Total: {total_codes} codes across {total_domains} domains in {len(hierarchy.themes)} theme(s)\n"
            
            formatted_parts.append(codebook_text)
        
        return "\n\n".join(formatted_parts)
    
    def _fix_missing_codes(self, reduced_structure: ConsolidatedHierarchy, batch_hierarchies: List[BatchHierarchy]) -> ConsolidatedHierarchy:
        """Programmatically add any missing codes to a Miscellaneous theme"""
        # Get all codes from the reduced structure
        found_codes = set()
        for theme in reduced_structure.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    found_codes.add(code.code_number)
        
        # Get all codes from the original batches
        all_codes = {}
        for batch in batch_hierarchies:
            for theme in batch.themes:
                for domain in theme.domains:
                    for code in domain.codes:
                        all_codes[code.code_number] = code
        
        # Find missing codes
        missing_code_numbers = set(all_codes.keys()) - found_codes
        
        if missing_code_numbers:
            self.verbose_reporter.stat_line(f"📌 Adding {len(missing_code_numbers)} missing codes to Miscellaneous theme")
            
            # Find or create Miscellaneous theme
            misc_theme = None
            for theme in reduced_structure.themes:
                if theme.theme_name.lower() in ["overige", "miscellaneous", "diversen"]:
                    misc_theme = theme
                    break
            
            if not misc_theme:
                # Create new Miscellaneous theme
                misc_theme = ThemeDefinition(
                    theme_name="Overige",
                    theme_concept="Diverse aspecten die niet in andere thema's passen",
                    domains=[]
                )
                reduced_structure.themes.append(misc_theme)
            
            # Find or create Miscellaneous domain
            misc_domain = None
            for domain in misc_theme.domains:
                if domain.domain_name.lower() in ["overige aspecten", "miscellaneous aspects", "diverse aspecten"]:
                    misc_domain = domain
                    break
            
            if not misc_domain:
                # Create new Miscellaneous domain
                misc_domain = DomainDefinition(
                    domain_name="Overige aspecten",
                    domain_description="Codes die niet goed in andere domeinen passen",
                    codes=[]
                )
                misc_theme.domains.append(misc_domain)
            
            # Add missing codes to the miscellaneous domain
            for code_num in sorted(missing_code_numbers):
                original_code = all_codes[code_num]
                misc_domain.codes.append(
                    CodeAssignment(
                        code_number=code_num,
                        code_name=original_code.code_name,
                        fit_rationale="Toegevoegd om volledigheid te garanderen"
                    )
                )
            
            self.verbose_reporter.stat_line(f"✅ All missing codes have been added to {misc_theme.theme_name} > {misc_domain.domain_name}")
        
        return reduced_structure
    
    async def _reduce_hierarchies(self, batch_hierarchies: List[BatchHierarchy]) -> ConsolidatedHierarchy:
        """Reduce stage: Merge multiple hierarchies into one consolidated hierarchy"""
        hierarchies_text = self._format_hierarchies_for_reduction(batch_hierarchies)
        
        # Count total codes in input batches for validation
        total_input_codes = sum(
            len(domain.codes) 
            for h in batch_hierarchies 
            for theme in h.themes 
            for domain in theme.domains
        )
        
        prompt = HIERARCHY_REDUCE_PROMPT.format(
            survey_question=self.var_lab,
            batch_hierarchies=hierarchies_text,
            total_codes=total_input_codes,
            language=DEFAULT_LANGUAGE
        )
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="hierarchy_reduce",
                utility_name="ThemeIdentifier",
                prompt_content=prompt,
                prompt_type="Hierarchy Consolidation"
            )
        
        max_attempts = 3
        best_response = None
        
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_config.get_model_for_stage("theme_synthesis"),
                    messages=[{"role": "user", "content": prompt}],
                    response_model=ConsolidatedHierarchy,
                    temperature=0.1 if attempt > 0 else 0.2,  # Lower temperature on retries
                    max_retries=3
                )
                
                # Validate that no codes were lost
                response_code_count = sum(
                    len(domain.codes) 
                    for theme in response.themes 
                    for domain in theme.domains
                )
                
                if response_code_count == total_input_codes:
                    # Log transformation information if available
                    if hasattr(response, 'transformation_notes'):
                        self.verbose_reporter.stat_line(f"✓ Tracked {len(getattr(response.transformation_notes, 'themes_merged', []))} theme mergers")
                    return response
                else:
                    # Keep the best response (most codes preserved)
                    if best_response is None or response_code_count > sum(
                        len(domain.codes) 
                        for theme in best_response.themes 
                        for domain in theme.domains
                    ):
                        best_response = response
                    
                    self.verbose_reporter.stat_line(
                        f"⚠️  Attempt {attempt + 1}: Lost {total_input_codes - response_code_count} codes. "
                        f"Expected {total_input_codes}, got {response_code_count}."
                    )
                    
            except Exception as e:
                self.verbose_reporter.stat_line(f"Error in hierarchy reduction attempt {attempt + 1}: {str(e)}")
        
        # If we reach here, no attempt succeeded - apply programmatic fix to best response
        if best_response is not None:
            self.verbose_reporter.stat_line(f"🔧 Applying programmatic fix to preserve all codes...")
            return self._fix_missing_codes(best_response, batch_hierarchies)
        else:
            # All attempts failed with exceptions - return empty structure
            self.verbose_reporter.stat_line(f"❌ All attempts failed with errors")
            return ConsolidatedHierarchy(
                themes=[],
                theme_transformations=[],
                domain_transformations=[]
            )
    
    def _build_final_codebook_structure(self, hierarchical_result: ConsolidatedHierarchy) -> Dict[str, Any]:
        """Build final structure with complete traceability - directly extract from hierarchy"""
        
        # Directly extract from hierarchy instead of relying on registry matching
        codebook_data = []
        
        for theme in hierarchical_result.themes:
            for domain in theme.domains:
                for code_assignment in domain.codes:
                    # Get the original definition from registry (fallback to code name if not found)
                    original_definition = self.code_registry.get(code_assignment.code_number, {}).get('definition', code_assignment.code_name)
                    
                    codebook_data.append({
                        'code_id': code_assignment.code_number,
                        'code': code_assignment.code_name,
                        'definition': original_definition,
                        'domain': domain.domain_name,
                        'theme': theme.theme_name
                    })
        
        # Sort by code_id to maintain original order
        codebook_data.sort(key=lambda x: x['code_id'])
        
        return {
            'codebook': codebook_data,
            'hierarchy': hierarchical_result
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
        
        # Map Stage: Parallel hierarchy creation
        batches = self._create_code_batches()
        self.verbose_reporter.stat_line(f"Created {len(batches)} batches for parallel hierarchy creation")
        
        hierarchy_tasks = []
        for i, batch in enumerate(batches, 1):
            task = self._create_hierarchy_for_batch(batch, i, len(batches))
            hierarchy_tasks.append(task)
        
        # Execute all hierarchy creation tasks in parallel
        self.verbose_reporter.stat_line("Starting parallel hierarchy creation (Map stage)...")
        batch_hierarchies = await asyncio.gather(*hierarchy_tasks)
        
        # Count total themes and domains from all batches
        total_batch_themes = sum(len(h.themes) for h in batch_hierarchies)
        total_batch_domains = sum(len(theme.domains) for h in batch_hierarchies for theme in h.themes)
        total_codes_in_batches = sum(
            len(domain.codes) 
            for h in batch_hierarchies 
            for theme in h.themes 
            for domain in theme.domains
        )
        self.verbose_reporter.stat_line(f"Created {total_batch_themes} themes across {total_batch_domains} domains from all batches")
        self.verbose_reporter.stat_line(f"Total codes in all batches: {total_codes_in_batches} (expected: {total_codes})")
        
        # Reduce Stage: Hierarchy consolidation
        self.verbose_reporter.stat_line("Starting hierarchy consolidation (Reduce stage)...")
        hierarchical_structure = await self._reduce_hierarchies(batch_hierarchies)
        
        # Validate completeness
        is_complete, missing_codes = self._validate_code_completeness(hierarchical_structure, total_codes)
        
        if not is_complete:
            self.verbose_reporter.stat_line(f"🔧 Applying programmatic fix for {len(missing_codes)} missing codes")
            # Print details of missing codes
            for num in sorted(missing_codes):
                if num in self.code_registry:
                    code_info = self.code_registry[num]
                    self.verbose_reporter.stat_line(f"   Missing Code {num}: {code_info['code_text'][:60]}...")
            
            # Apply fix
            hierarchical_structure = self._fix_missing_codes(hierarchical_structure, batch_hierarchies)
            
            # Validate again
            is_complete, still_missing = self._validate_code_completeness(hierarchical_structure, total_codes)
            if is_complete:
                self.verbose_reporter.stat_line(f"✅ All codes now present after programmatic fix")
            else:
                self.verbose_reporter.stat_line(f"❌ Still missing {len(still_missing)} codes after fix: {sorted(still_missing)}")
        else:
            self.verbose_reporter.stat_line(f"✅ All {total_codes} codes successfully consolidated")
        
        # Build final structure with traceability
        final_structure = self._build_final_codebook_structure(hierarchical_structure)
        
        elapsed_time = time.time() - start_time
        
        # Report transformation details if available
        if hasattr(hierarchical_structure, 'transformation_notes'):
            notes = hierarchical_structure.transformation_notes
            if hasattr(notes, 'themes_merged') and notes.themes_merged:
                self.verbose_reporter.stat_line(f"🔄 Theme mergers: {len(notes.themes_merged)}")
            if hasattr(notes, 'domains_merged') and notes.domains_merged:
                self.verbose_reporter.stat_line(f"🔄 Domain mergers: {len(notes.domains_merged)}")
        
        # Report results
        self.verbose_reporter.summary("HIERARCHICAL THEME IDENTIFICATION COMPLETE", {
            "Total codes": len(self.codebook),
            "Themes": len(hierarchical_structure.themes),
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
    
    def _validate_code_completeness(self, hierarchy: ConsolidatedHierarchy, expected_total: int) -> tuple[bool, set]:
        """Validate that all codes are present in the hierarchy"""
        found_codes = set()
        code_locations = {}  # Track where each code ended up
        
        for theme in hierarchy.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    if code.code_number in found_codes:
                        self.verbose_reporter.stat_line(
                            f"⚠️  Duplicate code {code.code_number} found in {theme.theme_name} > {domain.domain_name}"
                        )
                    found_codes.add(code.code_number)
                    code_locations[code.code_number] = {
                        'theme': theme.theme_name,
                        'domain': domain.domain_name
                    }
        
        expected_codes = set(range(1, expected_total + 1))
        missing_codes = expected_codes - found_codes
        
        if missing_codes:
            self.verbose_reporter.stat_line(f"❌ Missing {len(missing_codes)} codes: {sorted(missing_codes)}")
            return False, missing_codes
        
        if len(found_codes) > expected_total:
            extra_codes = found_codes - expected_codes
            self.verbose_reporter.stat_line(f"⚠️  Found unexpected codes: {sorted(extra_codes)}")
        
        return True, set()