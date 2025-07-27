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
        
        prompt = f"""
You are a qualitative researcher specializing in thematic analysis following Braun & Clarke (2006) methodology.
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
- If codes don't fit well elsewhere, include a "Miscellaneous" or "Other aspects" theme
- Focus on what the codes collectively reveal about responses to this survey question

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
        # Find or create miscellaneous theme
        misc_theme = None
        for theme in result.themes:
            if any(keyword in theme.theme_name.lower() for keyword in ["misc", "overig", "other", "diversen"]):
                misc_theme = theme
                break
        
        if not misc_theme:
            # Create new miscellaneous theme
            misc_theme = ThemeStructure(
                theme_name="Overige aspecten",
                theme_description="Codes die niet goed in andere thema's passen",
                codes=[]
            )
            result.themes.append(misc_theme)
        
        # Add missing codes
        for code_num in sorted(missing_codes):
            if code_num in self.code_registry:
                code_info = self.code_registry[code_num]
                misc_theme.codes.append(
                    CodeReference(
                        code_number=code_num,
                        code_name=code_info['code_name']
                    )
                )
        
        return result
    
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
                    temperature=0.1 if attempt > 0 else 0.2,
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