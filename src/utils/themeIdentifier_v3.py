import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import time
from typing import List, Dict, Any, Optional, Tuple
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
# PYDANTIC MODELS FOR ITERATIVE THEME IDENTIFICATION
# ============================================================================

class CodeReference(BaseModel):
    """Reference to a code in the theme hierarchy"""
    code_number: int = Field(description="Original code number from codebook")
    code_name: str = Field(description="Original code name")

class SingleThemeIdentification(BaseModel):
    """Result of identifying a single theme from available codes"""
    theme_identified: bool = Field(description="Whether a coherent theme was identified")
    theme_name: Optional[str] = Field(description="Descriptive theme name in target language")
    theme_description: Optional[str] = Field(description="Brief explanation of what unites these codes conceptually")
    assigned_codes: List[CodeReference] = Field(description="Codes that belong to this theme")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    rationale: str = Field(description="Explanation of theme identification or why no theme was found")
    
    @model_validator(mode='after')
    def validate_theme_consistency(self):
        """Ensure theme data is consistent"""
        if self.theme_identified:
            if not self.theme_name or not self.theme_description:
                raise ValueError("Theme name and description required when theme identified")
            if not self.assigned_codes:
                raise ValueError("At least one code must be assigned when theme identified")
        else:
            if self.assigned_codes:
                raise ValueError("No codes should be assigned when no theme identified")
        return self

class ThemeStructure(BaseModel):
    """Theme with codes following Braun & Clarke methodology"""
    theme_name: str = Field(description="Descriptive theme name in target language")
    theme_description: str = Field(description="Brief explanation of what unites these codes conceptually")
    codes: List[CodeReference] = Field(description="Codes that belong to this theme")
    iteration: int = Field(description="Iteration number when theme was identified")

# ============================================================================
# ITERATIVE THEME IDENTIFIER V3
# ============================================================================

class ThemeIdentifierV3:
    
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
        self.max_iterations = 20  # Prevent infinite loops
        self.min_codes_for_theme = 2  # Minimum codes to form a theme
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
    
    def _format_available_codes(self, available_codes: List[int]) -> str:
        """Format available codes as a numbered list"""
        formatted_codes = []
        for code_num in sorted(available_codes):
            code_info = self.code_registry[code_num]
            formatted_codes.append(f"{code_num}. {code_info['code_name']}")
        return "\n".join(formatted_codes)
    
    def _create_single_theme_prompt(self, available_codes: List[int], iteration: int, existing_themes: List[ThemeStructure]) -> str:
        """Create prompt for identifying a single theme"""
        codes_text = self._format_available_codes(available_codes)
        total_available = len(available_codes)
        
        existing_themes_text = ""
        if existing_themes:
            existing_themes_text = "\nALREADY IDENTIFIED THEMES:\n"
            for theme in existing_themes:
                existing_themes_text += f"- {theme.theme_name}: {theme.theme_description} ({len(theme.codes)} codes)\n"
        
        prompt = f"""You are a qualitative researcher specializing in thematic analysis following Braun & Clarke (2006) methodology.
Your task is to identify ONE coherent theme from the available codes below.

SURVEY QUESTION:
{self.var_lab}

ITERATION: {iteration}
{existing_themes_text}

AVAILABLE CODES ({total_available} remaining):
{codes_text}

YOUR TASK:
1. Examine these {total_available} codes carefully
2. Identify ONE coherent theme that unites multiple codes (minimum {self.min_codes_for_theme} codes)
3. Select ALL codes that belong to this theme
4. If no coherent theme can be formed from at least {self.min_codes_for_theme} codes, indicate this clearly

THEME IDENTIFICATION CRITERIA:
- Look for conceptual patterns and shared meanings
- Focus on salience and conceptual importance, not just frequency
- The theme should tell a meaningful story about the survey responses
- Each theme should be distinct from already identified themes
- Prioritize stronger, more coherent themes over weaker associations

INSTRUCTIONS:
- Identify exactly ONE theme (or indicate if none possible)
- Include ALL codes that fit this theme conceptually
- Theme name and description must be in {DEFAULT_LANGUAGE}
- Base analysis on code names, considering the survey context
- Be selective: only form themes with strong conceptual coherence

OUTPUT FORMAT (JSON):
{{
  "theme_identified": true/false,
  "theme_name": "[Descriptive theme name, or null if no theme]",
  "theme_description": "[Brief explanation of conceptual unity, or null if no theme]",
  "assigned_codes": [
    {{
      "code_number": [exact number from input],
      "code_name": "[exact code name from input]"
    }}
  ],
  "confidence": "high|medium|low",
  "rationale": "[Explanation of why this theme was identified or why no theme was possible]"
}}

CRITICAL: 
- Only use code numbers from the available list: {', '.join(str(c) for c in sorted(available_codes))}
- If you cannot identify a coherent theme, set theme_identified to false
- Focus on quality over quantity - better to have no theme than a weak one

Return ONLY the JSON object."""
        
        return prompt
    
    async def _identify_single_theme(self, available_codes: List[int], iteration: int, existing_themes: List[ThemeStructure]) -> SingleThemeIdentification:
        """Identify a single theme from available codes"""
        
        prompt = self._create_single_theme_prompt(available_codes, iteration, existing_themes)
        
        # Capture prompt for first iteration
        if iteration == 1 and self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="iterative_single_theme_identification",
                utility_name="ThemeIdentifierV3",
                prompt_content=prompt,
                prompt_type="Iterative Single Theme Identification"
            )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_stage("theme_synthesis"),
                messages=[{"role": "user", "content": prompt}],
                response_model=SingleThemeIdentification,
                temperature=0.1,  # Slight temperature for creativity
                max_retries=2
            )
            
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error in theme identification: {str(e)}")
            # Return no theme identified on error
            return SingleThemeIdentification(
                theme_identified=False,
                theme_name=None,
                theme_description=None,
                assigned_codes=[],
                confidence="low",
                rationale=f"Error during identification: {str(e)}"
            )
    
    def _validate_theme_codes(self, theme_result: SingleThemeIdentification, available_codes: List[int]) -> Tuple[bool, List[int]]:
        """Validate that assigned codes are from available codes"""
        if not theme_result.theme_identified:
            return True, []
        
        assigned_numbers = {code.code_number for code in theme_result.assigned_codes}
        available_set = set(available_codes)
        
        # Check if all assigned codes are available
        invalid_codes = assigned_numbers - available_set
        if invalid_codes:
            self.verbose_reporter.stat_line(f"⚠️  Invalid codes assigned: {sorted(invalid_codes)}")
            return False, list(invalid_codes)
        
        return True, []
    
    async def identify_themes_iteratively(self) -> Dict[str, Any]:
        """
        Main method: Identify themes iteratively, one at a time
        """
        
        self.verbose_reporter.section_header("ITERATIVE THEME IDENTIFICATION V3")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for theme identification")
            return {
                'codebook': [],
                'themes': [],
                'methodology': 'Iterative Braun & Clarke (2006)'
            }
        
        total_codes = len(self.codebook)
        available_codes = list(range(1, total_codes + 1))
        identified_themes = []
        iteration = 0
        
        self.verbose_reporter.stat_line(f"Starting with {total_codes} codes")
        
        # Iterative theme identification
        while available_codes and iteration < self.max_iterations:
            iteration += 1
            
            self.verbose_reporter.stat_line(f"\nIteration {iteration}: {len(available_codes)} codes remaining")
            
            # Identify one theme
            theme_result = await self._identify_single_theme(available_codes, iteration, identified_themes)
            
            # Validate result
            is_valid, invalid_codes = self._validate_theme_codes(theme_result, available_codes)
            
            if theme_result.theme_identified and is_valid and len(theme_result.assigned_codes) >= self.min_codes_for_theme:
                # Create theme structure
                theme = ThemeStructure(
                    theme_name=theme_result.theme_name,
                    theme_description=theme_result.theme_description,
                    codes=theme_result.assigned_codes,
                    iteration=iteration
                )
                identified_themes.append(theme)
                
                # Remove assigned codes from available pool
                assigned_numbers = {code.code_number for code in theme_result.assigned_codes}
                available_codes = [c for c in available_codes if c not in assigned_numbers]
                
                self.verbose_reporter.stat_line(
                    f"✅ Theme '{theme.theme_name}' identified with {len(theme.codes)} codes ({theme_result.confidence} confidence)"
                )
            else:
                # No valid theme identified
                if theme_result.theme_identified:
                    self.verbose_reporter.stat_line(
                        f"❌ Theme rejected: {theme_result.rationale}"
                    )
                else:
                    self.verbose_reporter.stat_line(
                        f"🛑 No more themes possible: {theme_result.rationale}"
                    )
                break
            
            # Safety check for minimum remaining codes
            if len(available_codes) < self.min_codes_for_theme:
                self.verbose_reporter.stat_line(
                    f"🛑 Only {len(available_codes)} codes remaining - below minimum for theme"
                )
                break
        
        # Handle remaining codes as miscellaneous
        if available_codes:
            misc_codes = [
                CodeReference(
                    code_number=code_num,
                    code_name=self.code_registry[code_num]['code_name']
                )
                for code_num in sorted(available_codes)
            ]
            
            misc_theme = ThemeStructure(
                theme_name="Overige aspecten",
                theme_description="Codes die niet in andere thema's passen",
                codes=misc_codes,
                iteration=iteration + 1
            )
            identified_themes.append(misc_theme)
            
            self.verbose_reporter.stat_line(
                f"📦 Created miscellaneous theme with {len(misc_codes)} remaining codes"
            )
        
        # Build final structure
        final_structure = self._build_final_codebook_structure(identified_themes)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        self.verbose_reporter.summary("ITERATIVE THEME IDENTIFICATION COMPLETE", {
            "Total codes": total_codes,
            "Themes identified": len(identified_themes),
            "Iterations": iteration,
            "Methodology": "Iterative Braun & Clarke (2006)",
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print detailed results if verbose
        if self.verbose and identified_themes:
            print("\nIdentified Themes (in order of identification):")
            for theme in identified_themes:
                print(f"\n  Iteration {theme.iteration} - {theme.theme_name}")
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
                # Get original definition from registry
                original_definition = self.code_registry.get(code_ref.code_number, {}).get('definition', code_ref.code_name)
                
                codebook_data.append({
                    'code_id': code_ref.code_number,
                    'code': code_ref.code_name,
                    'definition': original_definition,
                    'theme': theme.theme_name,
                    'theme_description': theme.theme_description,
                    'iteration_identified': theme.iteration
                })
        
        # Sort by code_id to maintain original order
        codebook_data.sort(key=lambda x: x['code_id'])
        
        return {
            'codebook': codebook_data,
            'themes': [
                {
                    'theme_name': theme.theme_name,
                    'theme_description': theme.theme_description,
                    'codes': theme.codes,
                    'iteration': theme.iteration
                }
                for theme in themes
            ],
            'methodology': 'Iterative Braun & Clarke (2006) - Single theme per iteration',
            'analysis_approach': 'Iterative theme identification to reduce cognitive load'
        }