import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class promptPrinter:
    """Captures and prints LLM prompts with storage and flexible output options."""
    
    def __init__(self, enabled: bool = True, print_realtime: bool = True):
        """
        Initialize promptPrinter.
        
        Args:
            enabled: Whether prompt printing is enabled
            print_realtime: Print prompts as they are captured (during pipeline)
        """
        self.enabled = enabled
        self.print_realtime = print_realtime
        self.prompts = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def capture_prompt(self, 
                      step_name: str,
                      utility_name: str,
                      prompt_content: str,
                      prompt_type: str = "main",
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Capture a prompt for storage and optional real-time printing.
        
        Args:
            step_name: Pipeline step (e.g., "preprocessing", "segmentation")
            utility_name: Utility class using the prompt (e.g., "SpellChecker")
            prompt_content: The actual prompt text
            prompt_type: Type of prompt (e.g., "main", "refinement", "validation")
            metadata: Additional metadata (e.g., model, temperature, var_lab)
        """
        if not self.enabled:
            return
            
        prompt_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "utility_name": utility_name,
            "prompt_type": prompt_type,
            "prompt_content": prompt_content,
            "metadata": metadata or {}
        }
        
        self.prompts.append(prompt_entry)
        
        if self.print_realtime:
            self._print_prompt(prompt_entry)
    
    def _print_prompt(self, prompt_entry: Dict[str, Any]) -> None:
        """Print a single prompt entry in detailed format."""
        print(f"\n{'='*80}")
        print(f"🤖 LLM PROMPT : {prompt_entry['step_name']}")
        print(f"{'='*80}")
        print(f"🔧 Utility: {prompt_entry['utility_name']}")
        
        # Print model info if available in metadata
        if 'model' in prompt_entry['metadata']:
            print(f"📊 Model: {prompt_entry['metadata']['model']}")
        
        print("📄 Prompt:")
        print(f"{'-'*80}")
        print(prompt_entry['prompt_content'])
        print(f"{'-'*80}")
    
    def print_all_prompts(self) -> None:
        """Print all captured prompts (useful for post-run review)."""
        if not self.enabled or not self.prompts:
            print("No prompts captured.")
            return
        
        print(f"\n{'#'*80}")
        print(f"# ALL CAPTURED PROMPTS - Session: {self.session_id}")
        print(f"# Total prompts: {len(self.prompts)}")
        print(f"{'#'*80}")
        
        for i, prompt_entry in enumerate(self.prompts, 1):
            print(f"\n[Prompt {i}/{len(self.prompts)}]")
            self._print_prompt(prompt_entry)
    
    def print_prompts_by_step(self, step_name: str) -> None:
        """Print all prompts for a specific pipeline step."""
        step_prompts = [p for p in self.prompts if p['step_name'] == step_name]
        
        if not step_prompts:
            print(f"No prompts found for step: {step_name}")
            return
        
        print(f"\n{'#'*80}")
        print(f"# PROMPTS FOR STEP: {step_name}")
        print(f"# Count: {len(step_prompts)}")
        print(f"{'#'*80}")
        
        for i, prompt_entry in enumerate(step_prompts, 1):
            print(f"\n[Step Prompt {i}/{len(step_prompts)}]")
            self._print_prompt(prompt_entry)
    
    def get_prompt_summary(self) -> Dict[str, Any]:
        """Get a summary of captured prompts."""
        if not self.prompts:
            return {"total": 0, "by_step": {}, "by_utility": {}}
        
        summary = {
            "total": len(self.prompts),
            "session_id": self.session_id,
            "by_step": {},
            "by_utility": {}
        }
        
        for prompt in self.prompts:
            step = prompt['step_name']
            utility = prompt['utility_name']
            
            summary['by_step'][step] = summary['by_step'].get(step, 0) + 1
            summary['by_utility'][utility] = summary['by_utility'].get(utility, 0) + 1
        
        return summary
    
    def print_summary(self) -> None:
        """Print a summary of all captured prompts."""
        summary = self.get_prompt_summary()
        
        print(f"\n{'📊 PROMPT CAPTURE SUMMARY'}")
        print(f"{'='*40}")
        print(f"Session ID: {summary.get('session_id', 'N/A')}")
        print(f"Total prompts captured: {summary['total']}")
        
        if summary['total'] > 0:
            print("\n📍 By Pipeline Step:")
            for step, count in summary['by_step'].items():
                print(f"  • {step}: {count}")
            
            print("\n🔧 By Utility:")
            for utility, count in summary['by_utility'].items():
                print(f"  • {utility}: {count}")
    
    def save_prompts(self, filepath: Optional[str] = None) -> str:
        """
        Save all prompts to a JSON file.
        
        Args:
            filepath: Optional custom filepath. 
            
        Returns:
            Path where prompts were saved
        """
        if not self.prompts:
            print("No prompts to save.")
            return ""
        
        if filepath:
            save_path = Path(filepath)
        else:
            save_path = Path(f"prompts_{self.session_id}.json")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "capture_time": datetime.now().isoformat(),
                "total_prompts": len(self.prompts),
                "summary": self.get_prompt_summary(),
                "prompts": self.prompts
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Prompts saved to: {save_path}")
        return str(save_path)