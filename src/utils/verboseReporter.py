import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import time
import random
import logging
from typing import List, Dict, Any, Tuple, Optional


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


class VerboseLoggingHandler(logging.Handler):
    """Custom logging handler that routes log messages through VerboseReporter."""
    
    def __init__(self, verbose_reporter: 'VerboseReporter', min_level: int = logging.WARNING):
        super().__init__()
        self.verbose_reporter = verbose_reporter
        self.min_level = min_level
        
    def emit(self, record: logging.LogRecord):
        """Handle a log record by routing it through VerboseReporter."""
        if record.levelno < self.min_level:
            return
            
        # Format the message
        msg = self.format(record)
        
        # Route based on level
        if record.levelno >= logging.ERROR:
            self.verbose_reporter.error(msg, source=record.name)
        elif record.levelno >= logging.WARNING:
            self.verbose_reporter.warning(msg, source=record.name)
        elif record.levelno >= logging.INFO:
            self.verbose_reporter.info(msg, source=record.name)
        else:
            self.verbose_reporter.debug(msg, source=record.name)


class VerboseReporter:
    """Handles formatted verbose output for pipeline operations with optional logging capture."""
    
    def __init__(self, enabled: bool = True, capture_logging: bool = True, 
                 log_level: int = logging.WARNING):
        """
        Initialize verbose reporter.
        
        Args:
            enabled: Whether verbose output is enabled
            capture_logging: Whether to capture and route logging messages
            log_level: Minimum logging level to capture (default: WARNING)
        """
        self.enabled = enabled
        self.start_time = None
        self.capture_logging = capture_logging
        self.log_level = log_level
        self._logging_handler = None
        self._original_handlers = {}
        
        if capture_logging:
            self._setup_logging_capture()
    
    def _setup_logging_capture(self):
        """Set up logging capture for common noisy libraries."""
        self._logging_handler = VerboseLoggingHandler(self, self.log_level)
        self._logging_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Libraries to capture
        libraries = [
            'openai', 'httpx', 'httpcore', 'tenacity', 
            'urllib3', 'asyncio', 'aiohttp'
        ]
        
        for lib_name in libraries:
            logger = logging.getLogger(lib_name)
            # Store original handlers
            self._original_handlers[lib_name] = logger.handlers.copy()
            # Clear existing handlers and add ours
            logger.handlers.clear()
            logger.addHandler(self._logging_handler)
            logger.setLevel(self.log_level)
    
    def restore_logging(self):
        """Restore original logging configuration."""
        if not self.capture_logging or not self._original_handlers:
            return
            
        for lib_name, handlers in self._original_handlers.items():
            logger = logging.getLogger(lib_name)
            logger.handlers.clear()
            for handler in handlers:
                logger.addHandler(handler)
    
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
    
    def warning(self, message: str, source: Optional[str] = None) -> None:
        """Print a warning message."""
        if not self.enabled:
            return
        source_text = f"[{source}] " if source else ""
        print(f"⚠️  {source_text}{message}")
    
    def error(self, message: str, source: Optional[str] = None) -> None:
        """Print an error message."""
        if not self.enabled:
            return
        source_text = f"[{source}] " if source else ""
        print(f"❌ {source_text}{message}")
    
    def info(self, message: str, source: Optional[str] = None) -> None:
        """Print an info message."""
        if not self.enabled:
            return
        source_text = f"[{source}] " if source else ""
        print(f"ℹ️  {source_text}{message}")
    
    def debug(self, message: str, source: Optional[str] = None) -> None:
        """Print a debug message."""
        if not self.enabled:
            return
        source_text = f"[{source}] " if source else ""
        print(f"🔍 {source_text}{message}")
    
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
            
        print("\n📋 Sample corrections:")
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
    
    def library_retry(self, attempt: int, max_attempts: int, reason: str, 
                     wait_time: Optional[float] = None) -> None:
        """Special method for retry attempts from libraries like tenacity."""
        if not self.enabled:
            return
        wait_text = f" (waiting {wait_time:.1f}s)" if wait_time else ""
        print(f"🔄 Retry {attempt}/{max_attempts}: {reason}{wait_text}")
    
    def rate_limit(self, service: str, wait_time: float) -> None:
        """Special method for rate limit notifications."""
        if not self.enabled:
            return
        print(f"⏳ Rate limit reached for {service}, waiting {wait_time:.1f}s...")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore logging."""
        self.restore_logging()
        return False