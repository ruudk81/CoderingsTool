import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import time
from typing import List, Dict, Any, Optional
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
# PYDANTIC MODELS FOR STRUCTURED BRAUN & CLARKE ANALYSIS
# ============================================================================

class ThemeAnalysis(BaseModel):
    """Single theme with detailed analysis following Braun & Clarke"""
    theme_name: str = Field(description="Descriptive name for the theme")
    concept: str = Field(description="Brief description of the underlying concept or narrative")
    codes: List[str] = Field(description="List of code names that belong to this theme")
    relationship: str = Field(description="Brief explanation of how these codes relate to each other and the overall theme")

class ReflectionAnalysis(BaseModel):
    """Reflection on the thematic analysis process"""
    broad_or_narrow_themes: str = Field(description="Discussion of any themes that seem too broad or too narrow")
    contradictions_or_unexpected_patterns: str = Field(description="Description of any contradictions or unexpected patterns")
    potential_subthemes: str = Field(description="Discussion of any need for subthemes within the main themes")
    unclassified_codes: str = Field(description="List of any codes that were not included in the proposed themes")

class BraunClarkeAnalysis(BaseModel):
    """Complete Braun & Clarke thematic analysis following specified structure"""
    initial_observations: List[str] = Field(description="Initial observations about the codes and patterns")
    suggested_themes: List[ThemeAnalysis] = Field(description="Array of identified themes with detailed analysis")
    reflection: ReflectionAnalysis = Field(description="Critical reflection on the analysis process and outcomes")

# ============================================================================
# BRAUN & CLARKE THEME IDENTIFIER V3
# ============================================================================

class ThemeIdentifierV3:
    """
    Theme identifier using exact Braun & Clarke methodology as specified by user
    Follows the 11-step process with deterministic analysis
    """
    
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
        
        # Configuration for deterministic analysis
        self.max_retries = 3
        self._initialize_code_registry()
    
    def _initialize_code_registry(self):
        """Create a registry of all codes for tracking and final structure building"""
        self.code_registry = {}
        for i, code in enumerate(self.codebook, 1):
            self.code_registry[code.code] = {
                'code_id': i,
                'code_name': code.code,
                'definition': code.definition
            }
    
    def _format_codes_for_analysis(self) -> str:
        """Format codes as simple list of names for analysis"""
        return "\n".join([code.code for code in self.codebook])
    
    def _create_braun_clarke_v3_prompt(self) -> str:
        """Create the exact Braun & Clarke prompt as specified by user"""
        codes_list = self._format_codes_for_analysis()
        
        prompt = f"""You are an expert qualitative researcher specializing in thematic analysis. Your task is to analyze a list of codes that will be given to you below in the <codes> tag and identify potential themes following the guidance of Braun and Clarke. The goal is to identify themes that help to answer the research question '{self.var_lab}'.
Please follow these steps outlined in the <instructions> tag carefully.

<instructions>
Step 1. Review the list of codes provided below in the <codes> tag below. These codes are being used to analyze survey responses from educational research context.
Step 2. Look for patterns and shared meanings among the codes. Consider how different codes might be combined based on underlying concepts or features of the data.
Step 3. Identify overarching narratives that might represent broader themes or sub-themes.
Step 4. Remember that themes don't simply "emerge" from the data. Actively construe relationships among the codes and examine how these relationships inform potential themes.
Step 5. Consider the importance and salience of potential themes. Remember, the number of codes supporting a theme is less important than whether the pattern communicates something meaningful that helps answer the research question(s). On that note, remember that the research question for this research is {self.var_lab}.
Step 6. Aim for themes that are distinctive yet coherent with the overall analysis. Themes may even be contradictory to each other.
Step 7. Be willing to let go of codes or potential themes that don't fit the overall analysis. Consider creating a "miscellaneous" category for codes that don't fit elsewhere.
Step 8. Strive for a balance in the number of themes − not so many that the analysis becomes unwieldy, but enough to fully explore the depth and breadth of the data.
Step 9. For each theme, prepare a structured description including the theme name, its underlying concept, associated codes, and how these codes relate to each other and the overall theme.
Step 10. Reflect on your analysis considering: themes that seem too broad or narrow, contradictions or unexpected patterns, need for subthemes, and codes that don't fit well into the current themes.
Step 11. Organize your analysis into a structured format with initial observations, an array of suggested themes (each as an object with name, concept, codes, and relationship), and your reflection.
</instructions>

Now that you have studied your instructions carefully, here is the list of codes to analyze to identify themes related to the research question "{self.var_lab}":
<codes>
{codes_list}
</codes>

Proceed with your expert analysis, explaining your reasoning at each step. Present your analysis in JSON format with the following structure:
{{
"initial_observations": [
"observation1"
],
"suggested_themes": [
{{
"theme_name": "Theme 1",
"concept": "Brief description of the underlying concept or narrative",
"codes": [
"Code 1"
],
"relationship": "Brief explanation of how these codes relate to each other and the overall theme"
}}
],
"reflection": {{
"broad_or_narrow_themes": "Discussion of any themes that seem too broad or too narrow",
"contradictions_or_unexpected_patterns": "Description of any contradictions or unexpected patterns",
"potential_subthemes": "Discussion of any need for subthemes within the main themes",
"unclassified_codes": "List of any codes that were not included in the proposed themes"
}}
}}

Use this JSON structure I have given you as a template. Expand on the template by adding as many observations, themes, and codes as necessary based on your analysis. Ensure that your response remains a valid JSON object. Do not include any text outside of this JSON structure.

Now that you have thoroughly read your task instructions, formatting instructions, and the codes to analyze, take a moment to gather your expert thoughts. Begin your analysis when you are ready."""
        
        return prompt
    
    def _validate_code_coverage(self, analysis: BraunClarkeAnalysis) -> tuple[bool, List[str]]:
        """Validate which codes are covered in the analysis"""
        all_code_names = {code.code for code in self.codebook}
        covered_codes = set()
        
        for theme in analysis.suggested_themes:
            for code_name in theme.codes:
                covered_codes.add(code_name)
        
        missing_codes = list(all_code_names - covered_codes)
        is_complete = len(missing_codes) == 0
        
        return is_complete, missing_codes
    
    async def identify_themes_braun_clarke_v3(self) -> Dict[str, Any]:
        """
        Main method: Identify themes using exact Braun & Clarke methodology as specified
        Uses deterministic analysis for reproducible results
        """
        
        self.verbose_reporter.section_header("BRAUN & CLARKE THEME IDENTIFICATION V3")
        start_time = time.time()
        
        # Check if codebook has codes
        if not self.codebook:
            self.verbose_reporter.stat_line("No codes available for theme identification")
            return {
                'analysis': None,
                'codebook': [],
                'methodology': 'Braun & Clarke (2006) - Exact Implementation'
            }
        
        total_codes = len(self.codebook)
        self.verbose_reporter.stat_line(f"Analyzing {total_codes} codes using exact Braun & Clarke methodology")
        
        # Create the analysis prompt
        prompt = self._create_braun_clarke_v3_prompt()
        
        # Capture prompt if printer is available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="braun_clarke_v3_theme_identification",
                utility_name="ThemeIdentifierV3",
                prompt_content=prompt,
                prompt_type="Braun & Clarke V3 - Exact Implementation"
            )
        
        # Attempt theme identification with deterministic settings
        best_analysis = None
        for attempt in range(self.max_retries):
            try:
                self.verbose_reporter.stat_line(f"Attempt {attempt + 1}/{self.max_retries}: Requesting deterministic analysis...")
                
                response = await self.client.chat.completions.create(
                    model=self.model_config.get_model_for_stage("theme_synthesis"),
                    messages=[{"role": "user", "content": prompt}],
                    response_model=BraunClarkeAnalysis,
                    temperature=0.0,  # Completely deterministic
                    max_retries=2
                )
                
                # Validate code coverage
                is_complete, missing_codes = self._validate_code_coverage(response)
                
                if is_complete:
                    self.verbose_reporter.stat_line(f"✅ All {total_codes} codes covered in {len(response.suggested_themes)} themes")
                    best_analysis = response
                    break
                else:
                    self.verbose_reporter.stat_line(f"⚠️  Missing {len(missing_codes)} codes: {missing_codes[:5]}{'...' if len(missing_codes) > 5 else ''}")
                    if best_analysis is None or len(missing_codes) < len(self._validate_code_coverage(best_analysis)[1]):
                        best_analysis = response
                        
            except Exception as e:
                self.verbose_reporter.stat_line(f"Error in attempt {attempt + 1}: {str(e)}")
                continue
        
        # Handle any missing codes by noting them in reflection
        if best_analysis is not None:
            is_complete, missing_codes = self._validate_code_coverage(best_analysis)
            if not is_complete:
                self.verbose_reporter.stat_line(f"⚠️  {len(missing_codes)} codes not explicitly assigned to themes")
                # Update reflection to note unclassified codes
                if missing_codes:
                    current_unclassified = best_analysis.reflection.unclassified_codes
                    if current_unclassified and current_unclassified.strip():
                        best_analysis.reflection.unclassified_codes = f"{current_unclassified}; Additional: {', '.join(missing_codes)}"
                    else:
                        best_analysis.reflection.unclassified_codes = f"Codes not explicitly assigned: {', '.join(missing_codes)}"
        else:
            # All attempts failed - create minimal fallback
            self.verbose_reporter.stat_line("❌ All attempts failed - creating fallback analysis")
            best_analysis = BraunClarkeAnalysis(
                initial_observations=["Analysis failed due to technical errors"],
                suggested_themes=[],
                reflection=ReflectionAnalysis(
                    broad_or_narrow_themes="Unable to complete analysis",
                    contradictions_or_unexpected_patterns="Technical failure prevented analysis",
                    potential_subthemes="Analysis incomplete",
                    unclassified_codes=f"All codes unclassified due to technical failure: {', '.join([code.code for code in self.codebook])}"
                )
            )
        
        # Build final structure
        final_structure = self._build_final_structure(best_analysis)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        self.verbose_reporter.summary("BRAUN & CLARKE V3 THEME IDENTIFICATION COMPLETE", {
            "Total codes": len(self.codebook),
            "Themes identified": len(best_analysis.suggested_themes),
            "Methodology": "Braun & Clarke (2006) - Exact Implementation",
            "Analysis complete": "Yes" if best_analysis.suggested_themes else "Failed",
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        # Print detailed results if verbose
        if self.verbose and best_analysis.suggested_themes:
            print("\nBraun & Clarke Analysis Results:")
            print(f"\nInitial Observations ({len(best_analysis.initial_observations)}):")
            for i, obs in enumerate(best_analysis.initial_observations, 1):
                print(f"  {i}. {obs}")
            
            print(f"\nSuggested Themes ({len(best_analysis.suggested_themes)}):")
            for i, theme in enumerate(best_analysis.suggested_themes, 1):
                print(f"\n  Theme {i}: {theme.theme_name}")
                print(f"    Concept: {theme.concept}")
                print(f"    Codes ({len(theme.codes)}): {', '.join(theme.codes)}")
                print(f"    Relationship: {theme.relationship}")
            
            print(f"\nReflection:")
            print(f"  Broad/Narrow: {best_analysis.reflection.broad_or_narrow_themes}")
            print(f"  Contradictions: {best_analysis.reflection.contradictions_or_unexpected_patterns}")
            print(f"  Subthemes: {best_analysis.reflection.potential_subthemes}")
            print(f"  Unclassified: {best_analysis.reflection.unclassified_codes}")
        
        return final_structure
    
    def _build_final_structure(self, analysis: BraunClarkeAnalysis) -> Dict[str, Any]:
        """Build final structure maintaining analysis format while providing codebook compatibility"""
        
        # Build codebook data for compatibility
        codebook_data = []
        
        for theme in analysis.suggested_themes:
            for code_name in theme.codes:
                if code_name in self.code_registry:
                    code_info = self.code_registry[code_name]
                    codebook_data.append({
                        'code_id': code_info['code_id'],
                        'code': code_name,
                        'definition': code_info['definition'],
                        'theme': theme.theme_name,
                        'theme_concept': theme.concept,
                        'theme_relationship': theme.relationship
                    })
        
        # Sort by code_id to maintain original order
        codebook_data.sort(key=lambda x: x['code_id'])
        
        return {
            'analysis': analysis,  # Complete Braun & Clarke analysis
            'codebook': codebook_data,  # Compatible format
            'methodology': 'Braun & Clarke (2006) - Exact Implementation',
            'analysis_approach': 'Deterministic 11-step process with structured reflection'
        }