"""
Verbose output reporter for pipeline operations.
Provides clean, formatted output with statistics and examples.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class VerboseReporter:
    """Handles formatted verbose output for pipeline operations."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_time = None
        
    def section_header(self, title: str, emoji: str = "📝") -> None:
        """Print a formatted section header."""
        if not self.enabled:
            return
        print(f"\n{emoji} {title.upper()}")
        print("=" * (len(title) + 4))
    
    def step_start(self, step_name: str, emoji: str = "🔄") -> None:
        """Start timing a processing step."""
        if not self.enabled:
            return
        print(f"\n{emoji} {step_name}")
        self.start_time = time.time()
    
    def step_complete(self, message: str = "", emoji: str = "✅") -> None:
        """Complete a step with timing info."""
        if not self.enabled:
            return
        elapsed = time.time() - self.start_time if self.start_time else 0
        timing = f" ({elapsed:.1f}s)" if elapsed > 0.1 else ""
        print(f"{emoji} {message}{timing}")
    
    def stat_line(self, message: str, bullet: str = "•") -> None:
        """Print a statistics line with bullet point."""
        if not self.enabled:
            return
        print(f"{bullet} {message}")
    
    def sample_list(self, title: str, samples: List[str], max_samples: int = 5) -> None:
        """Print a list of sample items."""
        if not self.enabled or not samples:
            return
        
        print(f"\n📋 {title}:")
        display_samples = random.sample(samples, min(len(samples), max_samples))
        for sample in display_samples:
            print(f'  "{sample}"')
    
    def correction_samples(self, corrections: List[Tuple[str, str]], max_samples: int = 5) -> None:
        """Print before/after correction samples."""
        if not self.enabled or not corrections:
            return
            
        print(f"\n📋 Sample corrections:")
        display_corrections = random.sample(corrections, min(len(corrections), max_samples))
        for before, after in display_corrections:
            print(f'  "{before}" → "{after}"')
    
    def summary(self, title: str, stats: Dict[str, Any], emoji: str = "📊") -> None:
        """Print a formatted summary section."""
        if not self.enabled:
            return
        
        print(f"\n{emoji} {title.upper()}")
        print("=" * (len(title) + 4))
        
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    def progress_line(self, current: int, total: int, operation: str = "") -> None:
        """Print a progress indicator."""
        if not self.enabled:
            return
        
        percentage = (current / total * 100) if total > 0 else 0
        operation_text = f" {operation}" if operation else ""
        print(f"Processing{operation_text}... {current}/{total} ({percentage:.1f}%)")


class ProcessingStats:
    """Helper class to collect and track processing statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.input_count = 0
        self.output_count = 0
        self.changes_made = 0
        self.items_changed = 0
        self.examples = []
        self.corrections = []
        self.start_time = None
        self.end_time = None
        
    def start_timing(self):
        """Start timing the operation."""
        self.start_time = time.time()
        
    def end_timing(self):
        """End timing the operation."""
        self.end_time = time.time()
        
    def get_duration(self) -> float:
        """Get the duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
        
    def add_example(self, example: str):
        """Add an example to the collection."""
        self.examples.append(example)
        
    def add_correction(self, before: str, after: str):
        """Add a before/after correction example."""
        self.corrections.append((before, after))
        
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.input_count == 0:
            return 0.0
        return (self.output_count / self.input_count) * 100