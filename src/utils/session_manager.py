"""
Optimized Session State Management for Streamlit Performance
Provides cached property access and efficient state validation
"""

import time
from typing import Optional, Any, Dict
from .bare_mode_utils import conditional_cache_resource, get_session_state, is_streamlit_context
import hashlib
import json


class SessionManager:
    """Optimized session state manager with caching and efficient validation"""
    
    def __init__(self):
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 10  # Cache valid for 10 seconds
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached value is still valid"""
        if key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[key] < self._cache_ttl
    
    def _get_cached(self, key: str, compute_fn):
        """Get cached value or compute and cache"""
        if key in self._cache and self._is_cache_valid(key):
            return self._cache[key]
        
        value = compute_fn()
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
        return value
    
    def invalidate_cache(self, key: str = None):
        """Invalidate specific key or entire cache"""
        if key:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    # Optimized property accessors
    @property
    def has_data(self) -> bool:
        """Check if data is loaded"""
        return self._get_cached('has_data', lambda: (
            hasattr(get_session_state(), 'data') and 
            get_session_state().data is not None and
            len(get_session_state().data) > 0
        ))
    
    @property
    def has_preprocessed_data(self) -> bool:
        """Check if preprocessing is complete"""
        return self._get_cached('has_preprocessed_data', lambda: (
            get_session_state().step >= 2 and
            hasattr(get_session_state(), 'preprocessed_data') and
            get_session_state().preprocessed_data is not None and
            len(get_session_state().preprocessed_data) > 0
        ))
    
    @property
    def has_quality_filtered_data(self) -> bool:
        """Check if quality filtering is complete"""
        return self._get_cached('has_quality_filtered_data', lambda: (
            get_session_state().step >= 3 and
            hasattr(get_session_state(), 'quality_filtered_data') and
            get_session_state().quality_filtered_data is not None and
            len(get_session_state().quality_filtered_data) > 0
        ))
    
    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are generated"""
        return self._get_cached('has_embeddings', lambda: (
            get_session_state().step >= 5 and
            hasattr(get_session_state(), 'embeddings_data') and
            get_session_state().embeddings_data is not None and
            len(get_session_state().embeddings_data) > 0
        ))
    
    @property
    def has_clusters(self) -> bool:
        """Check if clustering is complete"""
        return self._get_cached('has_clusters', lambda: (
            get_session_state().step >= 6 and
            hasattr(get_session_state(), 'cluster_results') and
            get_session_state().cluster_results is not None and
            len(get_session_state().cluster_results) > 0
        ))
    
    @property
    def data_size(self) -> int:
        """Get current data size efficiently"""
        return self._get_cached('data_size', lambda: (
            len(get_session_state().data) if self.has_data else 0
        ))
    
    @property
    def current_step(self) -> int:
        """Get current pipeline step"""
        return getattr(get_session_state(), 'step', 0)
    
    @property
    def spell_check_enabled(self) -> bool:
        """Check if spell checking is enabled in configuration"""
        return self._get_cached('spell_check_enabled', lambda: (
            hasattr(get_session_state(), 'spellcheck_config') and
            getattr(get_session_state().spellcheck_config, 'enabled', True)
        ))
    
    @property
    def embedding_provider(self) -> str:
        """Get selected embedding provider"""
        return self._get_cached('embedding_provider', lambda: (
            getattr(get_session_state(), 'embedding_provider', 'openai')
        ))
    
    def get_data_hash(self) -> Optional[str]:
        """Generate content hash of current data for caching"""
        if not self.has_data:
            return None
            
        return self._get_cached('data_hash', lambda: (
            self._compute_content_hash([
                getattr(item, 'response', str(item)) 
                for item in get_session_state().data[:100]  # Sample first 100 for performance
            ])
        ))
    
    def get_config_hash(self, config_obj) -> str:
        """Generate hash of configuration object"""
        config_dict = {}
        for attr in dir(config_obj):
            if not attr.startswith('_') and not callable(getattr(config_obj, attr)):
                config_dict[attr] = getattr(config_obj, attr)
        return self._compute_content_hash(config_dict)
    
    def _compute_content_hash(self, data) -> str:
        """Generate stable hash for content"""
        try:
            content = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return str(hash(str(data)))
    
    def should_skip_spell_check(self) -> bool:
        """Determine if spell checking should be skipped entirely"""
        return not self.spell_check_enabled
    
    def should_preload_next_step(self) -> bool:
        """Determine if we should preload resources for next pipeline step"""
        return self._get_cached('should_preload', lambda: (
            self.current_step >= 3 and  # Only after some progress
            self.data_size < 1000  # Only for smaller datasets to avoid memory issues
        ))
    
    def mark_step_complete(self, step: int):
        """Mark a pipeline step as complete and invalidate related caches"""
        get_session_state().step = max(get_session_state().step, step)
        
        # Invalidate caches that depend on step completion
        cache_keys_to_invalidate = [
            'has_preprocessed_data', 'has_quality_filtered_data', 
            'has_embeddings', 'has_clusters', 'should_preload'
        ]
        for key in cache_keys_to_invalidate:
            self.invalidate_cache(key)
    
    def cleanup_unused_resources(self):
        """Clean up session state resources that are no longer needed"""
        current_step = self.current_step
        
        # Clean up resources from previous steps to save memory
        if current_step > 3:
            # After preprocessing, clear raw text processing resources
            for key in ['text_normalizer', 'spell_checker_results']:
                if hasattr(get_session_state(), key):
                    delattr(get_session_state(), key)
        
        if current_step > 5:
            # After embeddings, clear intermediate processing results
            for key in ['quality_filter_details', 'preprocessing_stats']:
                if hasattr(get_session_state(), key):
                    delattr(get_session_state(), key)
        
        if current_step > 6:
            # After clustering, clear embedding computation details
            for key in ['embedding_batch_results', 'embedding_errors']:
                if hasattr(get_session_state(), key):
                    delattr(get_session_state(), key)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        import psutil
        import sys
        
        return {
            "current_step": self.current_step,
            "data_size": self.data_size,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "modules_loaded": len(sys.modules),
            "cache_entries": len(self._cache),
            "session_keys": len([k for k in dir(get_session_state()) if not k.startswith('_')])
        }


# Global session manager instance
@conditional_cache_resource
def get_session_manager():
    """Get cached session manager instance"""
    return SessionManager()