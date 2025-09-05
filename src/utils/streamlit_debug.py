"""
Streamlit Debug Features Module

Provides debugging capabilities matching pipeline.py features:
1. Verbose Report Display - Detailed processing stats and progress
2. First Prompt Capture - Display the first LLM prompt from each step  
3. Random Sample Results - Show sample outputs with regeneration

Integration with VerboseReporter, PromptPrinter, and sample generation.
"""

import streamlit as st
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import io
import sys
from contextlib import contextmanager

@dataclass
class DebugCapture:
    """Central debug data capture for pipeline steps"""
    
    # Control flags
    show_verbose: bool = False
    capture_prompts: bool = False
    show_samples: bool = False
    sample_count: int = 3
    sample_seed: int = 42
    
    # Captured data
    verbose_outputs: Dict[str, List[str]] = field(default_factory=dict)
    first_prompts: Dict[str, str] = field(default_factory=dict)  
    sample_results: Dict[str, Any] = field(default_factory=dict)
    step_timings: Dict[str, float] = field(default_factory=dict)
    step_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def clear_step_data(self, step_name: str):
        """Clear captured data for a specific step"""
        self.verbose_outputs.pop(step_name, None)
        self.first_prompts.pop(step_name, None)
        self.sample_results.pop(step_name, None)
        self.step_timings.pop(step_name, None)
        self.step_stats.pop(step_name, None)
    
    def has_debug_data(self, step_name: str) -> bool:
        """Check if any debug data exists for step"""
        return (step_name in self.verbose_outputs or 
                step_name in self.first_prompts or 
                step_name in self.sample_results)

class VerboseCapture:
    """Capture VerboseReporter output for Streamlit display"""
    
    def __init__(self, debug_capture: DebugCapture, step_name: str):
        self.debug_capture = debug_capture
        self.step_name = step_name
        self.outputs = []
        
    def capture_output(self, text: str):
        """Capture verbose output text"""
        if self.debug_capture.show_verbose:
            self.outputs.append(text)
            if self.step_name not in self.debug_capture.verbose_outputs:
                self.debug_capture.verbose_outputs[self.step_name] = []
            self.debug_capture.verbose_outputs[self.step_name].append(text)

class PromptCapture:
    """Capture first prompt from PromptPrinter for Streamlit display"""
    
    def __init__(self, debug_capture: DebugCapture, step_name: str):
        self.debug_capture = debug_capture
        self.step_name = step_name
        self.prompt_captured = False
        
    def capture_first_prompt(self, prompt: str, prompt_type: str = ""):
        """Capture the first prompt for this step"""
        if self.debug_capture.capture_prompts and not self.prompt_captured:
            prompt_key = f"{self.step_name}_{prompt_type}" if prompt_type else self.step_name
            self.debug_capture.first_prompts[prompt_key] = prompt
            self.prompt_captured = True

class SampleGenerator:
    """Generate random sample results for Streamlit display"""
    
    def __init__(self, debug_capture: DebugCapture, step_name: str):
        self.debug_capture = debug_capture
        self.step_name = step_name
        
    def generate_samples(self, data: Any, sample_type: str, sampler_func: callable) -> List[Any]:
        """Generate random samples using provided sampler function"""
        if not self.debug_capture.show_samples or not data:
            return []
            
        # Set random seed for reproducible samples (until user clicks regenerate)
        random.seed(self.debug_capture.sample_seed)
        
        try:
            samples = sampler_func(data, self.debug_capture.sample_count)
            
            # Store samples for display
            if self.step_name not in self.debug_capture.sample_results:
                self.debug_capture.sample_results[self.step_name] = {}
            self.debug_capture.sample_results[self.step_name][sample_type] = samples
            
            return samples
            
        except Exception as e:
            st.warning(f"Could not generate {sample_type} samples: {e}")
            return []

# Sample generation functions for each step type
class StepSamplers:
    """Standard sample generators for different pipeline steps"""
    
    @staticmethod
    def sample_spell_corrections(correction_examples: List, count: int) -> List[Dict[str, str]]:
        """Sample spell correction examples"""
        if not correction_examples or len(correction_examples) == 0:
            return []
        
        samples = random.sample(correction_examples, min(count, len(correction_examples)))
        return [{"original": sample[0], "corrected": sample[1]} for sample in samples]
    
    @staticmethod  
    def sample_idea_extractions(encoded_text: List, count: int) -> List[Dict[str, Any]]:
        """Sample idea extraction results"""
        if not encoded_text:
            return []
            
        samples = random.sample(encoded_text, min(count, len(encoded_text)))
        sample_data = []
        
        for item in samples:
            ideas = [segment.idea for segment in item.response_ideas] if hasattr(item, 'response_ideas') else []
            sample_data.append({
                "response": item.response,
                "idea_count": len(ideas),
                "ideas": ideas[:3]  # Show first 3 ideas
            })
        
        return sample_data
    
    @staticmethod
    def sample_cluster_contents(cluster_results: List, count: int) -> List[Dict[str, Any]]:
        """Sample cluster contents"""
        if not cluster_results:
            return []
            
        # Get unique cluster IDs
        cluster_ids = list(set([
            segment.initial_cluster 
            for result in cluster_results 
            for segment in result.response_ideas 
            if hasattr(segment, 'initial_cluster') and segment.initial_cluster is not None
        ]))
        
        if not cluster_ids:
            return []
            
        sampled_clusters = random.sample(cluster_ids, min(count, len(cluster_ids)))
        sample_data = []
        
        for cluster_id in sampled_clusters:
            # Get segments for this cluster
            cluster_segments = []
            for result in cluster_results:
                for segment in result.response_ideas:
                    if hasattr(segment, 'initial_cluster') and segment.initial_cluster == cluster_id:
                        cluster_segments.append(segment.idea)
            
            sample_data.append({
                "cluster_id": cluster_id,
                "segment_count": len(cluster_segments),
                "sample_segments": cluster_segments[:5]  # Show first 5 segments
            })
        
        return sample_data
    
    @staticmethod
    def sample_code_assignments(code_assigned_results: List, count: int) -> List[Dict[str, Any]]:
        """Sample code assignment results"""
        if not code_assigned_results:
            return []
            
        samples = random.sample(code_assigned_results, min(count, len(code_assigned_results)))
        sample_data = []
        
        for result in samples:
            assignments = []
            if hasattr(result, 'response_ideas'):
                for idea in result.response_ideas:
                    if hasattr(idea, 'assigned_codes') and idea.assigned_codes:
                        assignments.append({
                            "idea": idea.idea,
                            "codes": idea.assigned_codes,
                            "confidence": getattr(idea, 'assignment_confidence', None),
                            "rationale": getattr(idea, 'assignment_rationale', '')
                        })
            
            sample_data.append({
                "respondent_id": result.respondent_id,
                "response": result.response,
                "assignment_count": len(assignments),
                "assignments": assignments[:2]  # Show first 2 assignments
            })
        
        return sample_data

# Streamlit display functions
class DebugDisplayer:
    """Handle debug information display in Streamlit interface"""
    
    @staticmethod
    def display_verbose_reports(debug_capture: DebugCapture):
        """Display verbose reports for all steps with data"""
        if not debug_capture.show_verbose or not debug_capture.verbose_outputs:
            return
            
        st.subheader("🔍 Verbose Reports")
        
        for step_name, outputs in debug_capture.verbose_outputs.items():
            if outputs:
                with st.expander(f"📋 {step_name.replace('_', ' ').title()} Verbose Report"):
                    for output in outputs:
                        st.text(output)
    
    @staticmethod
    def display_captured_prompts(debug_capture: DebugCapture):
        """Display first captured prompt for each step"""
        if not debug_capture.capture_prompts or not debug_capture.first_prompts:
            return
            
        st.subheader("📝 Captured Prompts")
        
        for step_prompt, prompt_text in debug_capture.first_prompts.items():
            if prompt_text:
                with st.expander(f"🤖 {step_prompt.replace('_', ' ').title()} - First Prompt"):
                    st.code(prompt_text, language="text")
    
    @staticmethod
    def display_sample_results(debug_capture: DebugCapture):
        """Display sample results for all steps with data"""
        if not debug_capture.show_samples or not debug_capture.sample_results:
            return
            
        st.subheader("🎲 Sample Results")
        
        # Add regenerate button at the top
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Generate New Samples"):
                debug_capture.sample_seed = random.randint(1, 10000)
                st.rerun()
        with col2:
            st.write(f"*Sample seed: {debug_capture.sample_seed}*")
        
        for step_name, step_samples in debug_capture.sample_results.items():
            if step_samples:
                with st.expander(f"🎯 {step_name.replace('_', ' ').title()} Sample Results"):
                    DebugDisplayer._display_step_samples(step_name, step_samples)
    
    @staticmethod
    def _display_step_samples(step_name: str, step_samples: Dict[str, Any]):
        """Display samples for a specific step"""
        for sample_type, samples in step_samples.items():
            st.write(f"**{sample_type.replace('_', ' ').title()}:**")
            
            if sample_type == "spell_corrections":
                for i, sample in enumerate(samples, 1):
                    st.write(f"  {i}. `{sample['original']}` → `{sample['corrected']}`")
                    
            elif sample_type == "idea_extractions":
                for i, sample in enumerate(samples, 1):
                    st.write(f"  **{i}. Response:** {sample['response'][:100]}...")
                    st.write(f"     **Ideas ({sample['idea_count']}):** {', '.join(sample['ideas'])}")
                    
            elif sample_type == "cluster_contents":
                for i, sample in enumerate(samples, 1):
                    st.write(f"  **{i}. Cluster {sample['cluster_id']}** ({sample['segment_count']} segments):")
                    for segment in sample['sample_segments']:
                        st.write(f"     - {segment}")
                        
            elif sample_type == "code_assignments":
                for i, sample in enumerate(samples, 1):
                    st.write(f"  **{i}. ID {sample['respondent_id']}:** {sample['response'][:80]}...")
                    for assignment in sample['assignments']:
                        st.write(f"     - *{assignment['idea'][:50]}...* → **{', '.join(assignment['codes'])}**")
            
            st.write("")  # Add spacing

@contextmanager
def capture_stdout():
    """Context manager to capture stdout for verbose output"""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout

def create_debug_capture_from_session() -> Optional[DebugCapture]:
    """Create DebugCapture from Streamlit session state"""
    if not hasattr(st.session_state, 'debug_enabled'):
        return None
        
    return DebugCapture(
        show_verbose=getattr(st.session_state, 'debug_verbose', False),
        capture_prompts=getattr(st.session_state, 'debug_prompts', False),
        show_samples=getattr(st.session_state, 'debug_samples', False),
        sample_count=getattr(st.session_state, 'debug_sample_count', 3),
        sample_seed=getattr(st.session_state, 'debug_sample_seed', 42)
    )

def display_debug_controls():
    """Display debug control widgets in Streamlit sidebar"""
    st.sidebar.subheader("🔍 Debug Options")
    
    # Enable/disable debug entirely
    debug_enabled = st.sidebar.checkbox(
        "Enable Debug Features", 
        value=getattr(st.session_state, 'debug_enabled', False),
        key='debug_enabled'
    )
    
    if debug_enabled:
        # Individual debug toggles
        st.session_state.debug_verbose = st.sidebar.checkbox(
            "📋 Show Verbose Reports", 
            value=getattr(st.session_state, 'debug_verbose', False)
        )
        
        st.session_state.debug_prompts = st.sidebar.checkbox(
            "🤖 Capture First Prompts", 
            value=getattr(st.session_state, 'debug_prompts', False)
        )
        
        st.session_state.debug_samples = st.sidebar.checkbox(
            "🎲 Show Sample Results", 
            value=getattr(st.session_state, 'debug_samples', False)
        )
        
        if st.session_state.debug_samples:
            st.session_state.debug_sample_count = st.sidebar.slider(
                "Sample Count", 
                min_value=1, max_value=10, 
                value=getattr(st.session_state, 'debug_sample_count', 3)
            )
    
    return debug_enabled

def display_all_debug_info(debug_capture: DebugCapture):
    """Display all captured debug information"""
    if not debug_capture:
        return
        
    # Only show debug section if there's any data
    if (debug_capture.verbose_outputs or 
        debug_capture.first_prompts or 
        debug_capture.sample_results):
        
        st.markdown("---")
        st.header("🔍 Debug Information")
        
        # Display each type of debug info
        DebugDisplayer.display_verbose_reports(debug_capture)
        DebugDisplayer.display_captured_prompts(debug_capture)
        DebugDisplayer.display_sample_results(debug_capture)