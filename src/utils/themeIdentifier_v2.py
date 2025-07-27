import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import time
from typing import List, Dict, Any
import instructor
from openai import AsyncOpenAI

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
# PYDANTIC MODELS FOR BRAUN & CLARKE THEME IDENTIFICATION
# ============================================================================

class CodeReference(BaseModel):
    """Reference to a code in the theme hierarchy"""
    code_number: int = Field(description="Original code number from codebook")
    code_name: str = Field(description="Original code name")

class ThemeStructure(BaseModel):
    """Theme with codes following Braun & Clarke methodology"""
    theme_name: str = Field(description="Descriptive theme name in target language")
    theme_description: str = Field(description="Brief explanation of what unites these codes conceptually")
    codes: List[CodeReference] = Field(description="Codes that belong to this theme")
    
    @model_validator(mode='after')
    def validate_theme_has_codes(self):
        """Ensure each theme has at least one code"""
        if not self.codes:
            raise ValueError("Each theme must contain at least one code")
        return self

class BraunClarkeCodebook(BaseModel):
    """Complete codebook following Braun & Clarke methodology"""
    themes: List[ThemeStructure] = Field(description="All identified themes")
    methodology_notes: str = Field(description="Brief note about the thematic analysis approach used")
    
    @model_validator(mode='after')
    def validate_completeness(self):
        """Validate that all codes are included and no duplicates exist"""
        found_codes = set()
        for theme in self.themes:
            for code in theme.codes:
                if code.code_number in found_codes:
                    raise ValueError(f"Code {code.code_number} appears multiple times")
                found_codes.add(code.code_number)
        
        self._found_codes = found_codes
        return self

# ============================================================================
# BRAUN & CLARKE THEME IDENTIFIER
# ============================================================================

class ThemeIdentifierV2:
    
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
        
        # Configuration for Braun & Clarke approach
        self.max_retries = 3
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
    
    def _format_codes_for_analysis(self) -> str:
        """Format all codes as a simple numbered list of names only"""
        formatted_codes = []
        for i, code in enumerate(self.codebook, 1):
            formatted_codes.append(f"{i}. {code.code}")
        return "\n".join(formatted_codes)
    
    def _create_braun_clarke_prompt(self) -> str:
        """Create the Braun & Clarke themed analysis prompt"""
        codes_text = self._format_codes_for_analysis()
        total_codes = len(self.codebook)
        
        prompt = f"""You are a qualitative researcher specializing in thematic analysis following Braun & Clarke (2006) methodology.
Your task is to identify themes by analyzing code names that emerged from survey responses.

SURVEY QUESTION:
{self.var_lab}

CODES TO ANALYZE (names only):
{codes_text}

BRAUN & CLARKE METHODOLOGY:
1. **Familiarization**: These {total_codes} codes represent patterns identified in survey responses
2. **Pattern Recognition**: Look for repeated meanings and conceptual connections between codes
3. **Theme Construction**: Actively construct themes that capture coherent patterns of meaning
4. **Salience over Frequency**: Focus on conceptual importance, not just how many codes fit
5. **Coherent Narrative**: Each theme should tell a meaningful story about the data

INSTRUCTIONS:
- Identify themes that capture the main conceptual patterns
- Each code must be assigned to exactly one theme
- Themes should represent coherent patterns of meaning related to the survey question
- Focus on what the codes collectively reveal about responses to this survey question
- It's perfectly acceptable to include a "Miscellaneous" theme for codes that don't clearly fit other themes - this will be refined in a second analysis stage

CRITICAL REQUIREMENTS:
1. ALL {total_codes} codes must appear exactly once in your output
2. Use these exact code numbers: {', '.join(str(i) for i in range(1, total_codes + 1))}
3. Theme names and descriptions should be in {DEFAULT_LANGUAGE}
4. Base analysis solely on code names, not inferred definitions

OUTPUT FORMAT (JSON):
{{
  "themes": [
    {{
      "theme_name": "[Descriptive theme name capturing the conceptual pattern]",
      "theme_description": "[Brief explanation of what unites these codes conceptually]",
      "codes": [
        {{
          "code_number": [exact number from input],
          "code_name": "[exact code name from input]"
        }}
      ]
    }}
  ],
  "methodology_notes": "Thematic analysis following Braun & Clarke (2006) methodology, analyzing code names for conceptual patterns related to: {self.var_lab}"
}}

VALIDATION CHECKLIST:
- Are all {total_codes} codes included exactly once?
- Do theme names capture meaningful conceptual patterns?
- Are themes distinct yet coherent?
- Do themes relate meaningfully to the survey question?

Return ONLY the JSON object with all content in {DEFAULT_LANGUAGE}."""
        
        return prompt
    
    def _validate_completeness(self, result: BraunClarkeCodebook) -> tuple[bool, set]:
        """Validate that all codes are present and no duplicates exist"""
        expected_codes = set(range(1, len(self.codebook) + 1))
        found_codes = getattr(result, '_found_codes', set())
        
        if not found_codes:
            # Manual extraction if validator didn't run
            found_codes = set()
            for theme in result.themes:
                for code in theme.codes:
                    found_codes.add(code.code_number)
        
        missing_codes = expected_codes - found_codes
        is_complete = len(missing_codes) == 0
        
        return is_complete, missing_codes
    
    def _add_missing_codes_to_miscellaneous(self, result: BraunClarkeCodebook, missing_codes: set) -> BraunClarkeCodebook:
        """Add missing codes to a miscellaneous theme"""
        # Find existing miscellaneous theme
        misc_theme = None
        for theme in result.themes:
            if any(keyword in theme.theme_name.lower() for keyword in ["misc", "overig", "other", "diversen"]):
                misc_theme = theme
                break
        
        # Prepare missing codes as CodeReference objects
        missing_code_refs = []
        for code_num in sorted(missing_codes):
            if code_num in self.code_registry:
                code_info = self.code_registry[code_num]
                missing_code_refs.append(
                    CodeReference(
                        code_number=code_num,
                        code_name=code_info['code_name']
                    )
                )
        
        if not misc_theme:
            # Create new miscellaneous theme with the missing codes
            misc_theme = ThemeStructure(
                theme_name="Overige aspecten",
                theme_description="Codes die niet goed in andere thema's passen",
                codes=missing_code_refs
            )
            result.themes.append(misc_theme)
        else:
            # Add missing codes to existing miscellaneous theme
            misc_theme.codes.extend(missing_code_refs)
        
        return result
    
    def _create_miscellaneous_refinement_prompt(self, misc_codes: List[CodeReference], existing_themes: List[ThemeStructure]) -> str:
        """Create prompt for analyzing miscellaneous codes in stage 2"""
        misc_codes_text = "\n".join([f"{code.code_number}. {code.code_name}" for code in misc_codes])
        
        existing_themes_text = ""
        for theme in existing_themes:
            if any(keyword in theme.theme_name.lower() for keyword in ["misc", "overig", "other", "diversen"]):
                continue  # Skip miscellaneous theme itself
            existing_themes_text += f"- {theme.theme_name}: {theme.theme_description}\n"
        
        prompt = f"""You are a qualitative researcher specializing in thematic analysis following Braun & Clarke (2006) methodology.
In Stage 1, clear themes were identified, but some codes were placed in "Miscellaneous". Your task is to analyze these miscellaneous codes to either:
1. Create new meaningful themes from groups of these codes, OR
2. Assign individual codes to existing themes where they fit better

SURVEY QUESTION:
{self.var_lab}

EXISTING THEMES (from Stage 1):
{existing_themes_text}

MISCELLANEOUS CODES TO ANALYZE:
{misc_codes_text}

GUIDANCE FOR BETTER PLACEMENT:
- Physical activity/movement codes → likely fit "Gezondheid en Welzijn" themes
- Monitoring/compliance/implementation codes → likely fit "Organisatie en Implementatie" themes  
- Prevention/education codes → likely fit "Gezondheid en Welzijn" or "Bewustwording" themes
- Organizational activity codes → likely fit "Organisatie en Implementatie" themes
- Look for conceptual overlap rather than exact keyword matching

INSTRUCTIONS:
- Look for patterns among these {len(misc_codes)} miscellaneous codes
- Consider if any groups of 2+ codes could form new coherent themes
- Consider if individual codes actually belong in existing themes
- PRIORITIZE reassigning codes to existing themes where they conceptually fit
- Only create new themes if you find clear patterns among multiple codes
- Only if codes don't fit any theme, include a "Miscellaneous" theme. But realize that your reputation as a qualitative researcher is on the line for creating meaningful, coherent themes that minimize miscellaneous categorization
- Focus on creating meaningful, defensible groupings
- GOAL: Assign ALL miscellaneous codes to meaningful themes (existing or new)

OUTPUT FORMAT (JSON):
{{
  "new_themes": [
    {{
      "theme_name": "[Name for new theme]",
      "theme_description": "[What unites these codes]",
      "codes": [
        {{
          "code_number": [number],
          "code_name": "[name]"
        }}
      ]
    }}
  ],
  "reassignments": [
    {{
      "code_number": [number],
      "code_name": "[name]",
      "target_theme": "[existing theme name to assign to]",
      "rationale": "[why this code fits better here]"
    }}
  ],
  "remaining_miscellaneous": [
    {{
      "code_number": [number],
      "code_name": "[name]"
    }}
  ]
}}

CRITICAL VALIDATION:
Before submitting, verify that EVERY miscellaneous code ({len(misc_codes)} total) appears in either:
- new_themes, OR  
- reassignments, OR
- remaining_miscellaneous (only as last resort)

The sum of codes across all three categories must equal {len(misc_codes)}.

Return ONLY the JSON object with all content in {DEFAULT_LANGUAGE}."""
        
        return prompt
    
    async def _refine_miscellaneous_codes(self, initial_result: BraunClarkeCodebook) -> BraunClarkeCodebook:
        """Stage 2: Analyze miscellaneous codes for better themes or assignments"""
        
        # Find miscellaneous theme and extract its codes
        misc_theme = None
        misc_codes = []
        for theme in initial_result.themes:
            if any(keyword in theme.theme_name.lower() for keyword in ["misc", "overig", "other", "diversen"]):
                misc_theme = theme
                misc_codes = theme.codes.copy()
                break
        
        if not misc_theme or len(misc_codes) < 2:
            # No miscellaneous theme or too few codes to refine
            self.verbose_reporter.stat_line("No miscellaneous refinement needed")
            return initial_result
        
        self.verbose_reporter.stat_line(f"Stage 2: Refining {len(misc_codes)} miscellaneous codes...")
        
        # Create refinement prompt
        other_themes = [t for t in initial_result.themes if t != misc_theme]
        refinement_prompt = self._create_miscellaneous_refinement_prompt(misc_codes, other_themes)
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="braun_clarke_refinement",
                utility_name="ThemeIdentifierV2",
                prompt_content=refinement_prompt,
                prompt_type="Stage 2: Miscellaneous Refinement"
            )
        
        try:
            # Simple refinement structure
            class MiscellaneousRefinement(BaseModel):
                new_themes: List[ThemeStructure] = Field(description="New themes created from miscellaneous codes")
                reassignments: List[Dict[str, Any]] = Field(description="Codes to reassign to existing themes")
                remaining_miscellaneous: List[CodeReference] = Field(description="Codes that remain miscellaneous")
            
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("theme_synthesis"),
                messages=[{"role": "user", "content": refinement_prompt}],
                response_model=MiscellaneousRefinement,
                temperature=0.0,
                max_retries=2
            )
            
            # Apply refinements to create updated result
            updated_themes = other_themes.copy()
            
            # Add new themes
            for new_theme in response.new_themes:
                updated_themes.append(new_theme)
                self.verbose_reporter.stat_line(f"✓ Created new theme: '{new_theme.theme_name}' with {len(new_theme.codes)} codes")
            
            # Apply reassignments
            for reassignment in response.reassignments:
                target_theme_name = reassignment['target_theme']
                code_to_reassign = CodeReference(
                    code_number=reassignment['code_number'],
                    code_name=reassignment['code_name']
                )
                
                # Find target theme and add code
                for theme in updated_themes:
                    if theme.theme_name == target_theme_name:
                        theme.codes.append(code_to_reassign)
                        self.verbose_reporter.stat_line(f"✓ Reassigned Code {code_to_reassign.code_number} to '{target_theme_name}'")
                        break
            
            # Handle remaining miscellaneous codes
            if response.remaining_miscellaneous:
                remaining_misc_theme = ThemeStructure(
                    theme_name="Overige aspecten",
                    theme_description="Codes die niet in andere thema's passen",
                    codes=response.remaining_miscellaneous
                )
                updated_themes.append(remaining_misc_theme)
                self.verbose_reporter.stat_line(f"✓ {len(response.remaining_miscellaneous)} codes remain in miscellaneous")
            else:
                self.verbose_reporter.stat_line("✓ All miscellaneous codes successfully categorized!")
            
            # Create updated result
            return BraunClarkeCodebook(
                themes=updated_themes,
                methodology_notes=f"{initial_result.methodology_notes} - Enhanced with two-stage refinement"
            )
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Stage 2 refinement failed: {str(e)}")
            return initial_result
    
    async def identify_themes_braun_clarke(self) -> Dict[str, Any]:
        """
        Main method: Identify themes using Braun & Clarke methodology
        Analyzes only code names for better LLM performance
        """
        
        self.verbose_reporter.section_header("BRAUN & CLARKE THEME IDENTIFICATION")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for theme identification")
            return {
                'codebook': [],
                'themes': [],
                'methodology': 'Braun & Clarke (2006)'
            }
        
        total_codes = len(self.codebook)
        self.verbose_reporter.stat_line(f"Analyzing {total_codes} code names using Braun & Clarke methodology")
        
        # Create the analysis prompt
        prompt = self._create_braun_clarke_prompt()
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="braun_clarke_theme_identification",
                utility_name="ThemeIdentifierV2",
                prompt_content=prompt,
                prompt_type="Braun & Clarke Thematic Analysis"
            )
        
        # Attempt theme identification with retries
        best_result = None
        for attempt in range(self.max_retries):
            try:
                self.verbose_reporter.stat_line(f"Attempt {attempt + 1}/{self.max_retries}: Requesting theme identification...")
                
                response = await self.client.chat.completions.create(
                    model=self.model_config.get_model_for_stage("theme_synthesis"),
                    messages=[{"role": "user", "content": prompt}],
                    response_model=BraunClarkeCodebook,
                    temperature=0.0,  # Deterministic for reproducible results
                    max_retries=2
                )
                
                # Validate completeness
                is_complete, missing_codes = self._validate_completeness(response)
                
                if is_complete:
                    self.verbose_reporter.stat_line(f"✅ All {total_codes} codes successfully assigned to {len(response.themes)} themes")
                    best_result = response
                    break
                else:
                    self.verbose_reporter.stat_line(f"⚠️  Missing {len(missing_codes)} codes: {sorted(missing_codes)}")
                    if best_result is None or len(missing_codes) < len(self._validate_completeness(best_result)[1]):
                        best_result = response
                        
            except Exception as e:
                self.verbose_reporter.stat_line(f"Error in attempt {attempt + 1}: {str(e)}")
                continue
        
        # Apply fix for missing codes if needed
        if best_result is not None:
            is_complete, missing_codes = self._validate_completeness(best_result)
            if not is_complete:
                self.verbose_reporter.stat_line(f"🔧 Adding {len(missing_codes)} missing codes to miscellaneous theme")
                best_result = self._add_missing_codes_to_miscellaneous(best_result, missing_codes)
                
                # Validate again
                is_complete, still_missing = self._validate_completeness(best_result)
                if is_complete:
                    self.verbose_reporter.stat_line("✅ All codes now present after adding to miscellaneous theme")
                else:
                    self.verbose_reporter.stat_line(f"❌ Still missing {len(still_missing)} codes: {sorted(still_missing)}")
            
            # Stage 2: Refine miscellaneous codes
            self.verbose_reporter.stat_line("Starting Stage 2: Miscellaneous code refinement...")
            best_result = await self._refine_miscellaneous_codes(best_result)
        else:
            # All attempts failed - create fallback structure
            self.verbose_reporter.stat_line("❌ All attempts failed - creating fallback structure")
            codes = [
                CodeReference(code_number=i, code_name=code.code)
                for i, code in enumerate(self.codebook, 1)
            ]
            best_result = BraunClarkeCodebook(
                themes=[
                    ThemeStructure(
                        theme_name="Alle aspecten",
                        theme_description="Alle codes gegroepeerd vanwege technische problemen",
                        codes=codes
                    )
                ],
                methodology_notes="Fallback structure due to analysis errors"
            )
        
        # Build final codebook structure
        final_structure = self._build_final_codebook_structure(best_result)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        self.verbose_reporter.summary("BRAUN & CLARKE THEME IDENTIFICATION COMPLETE", {
            "Total codes": len(self.codebook),
            "Themes identified": len(best_result.themes),
            "Methodology": "Braun & Clarke (2006)",
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print detailed results if verbose
        if self.verbose and best_result.themes:
            print("\nIdentified Themes:")
            for i, theme in enumerate(best_result.themes, 1):
                print(f"\n  Theme {i}: {theme.theme_name}")
                print(f"    Description: {theme.theme_description}")
                print(f"    Codes ({len(theme.codes)}): {', '.join([f'{c.code_number}. {c.code_name[:40]}...' if len(c.code_name) > 40 else f'{c.code_number}. {c.code_name}' for c in theme.codes])}")
        
        return final_structure
    
    def _build_final_codebook_structure(self, braun_clarke_result: BraunClarkeCodebook) -> Dict[str, Any]:
        """Build final structure with theme assignments"""
        codebook_data = []
        
        for theme in braun_clarke_result.themes:
            for code_ref in theme.codes:
                # Get original definition from registry
                original_definition = self.code_registry.get(code_ref.code_number, {}).get('definition', code_ref.code_name)
                
                codebook_data.append({
                    'code_id': code_ref.code_number,
                    'code': code_ref.code_name,
                    'definition': original_definition,
                    'theme': theme.theme_name,
                    'theme_description': theme.theme_description
                })
        
        # Sort by code_id to maintain original order
        codebook_data.sort(key=lambda x: x['code_id'])
        
        return {
            'codebook': codebook_data,
            'themes': braun_clarke_result.themes,
            'methodology': braun_clarke_result.methodology_notes,
            'analysis_approach': 'Braun & Clarke (2006) - Code names only'
        }