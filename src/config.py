"""Consolidated configuration for the CoderingsTool pipeline"""

import os
import sqlite3
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# File handling
ALLOWED_EXTENSIONS = ['.sav']
MAX_FILE_SIZE_MB = 50

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
CONTEXT_WINDOW = 4000
MAX_OUTPUT_TOKENS = 1000

# Preprocessing settings
BATCH_SIZE = 100

# Hunspell path detection
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'utils':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'hunspell'))
elif os.path.basename(current_dir) == 'modules':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'hunspell'))
elif os.path.basename(current_dir) == 'src':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', 'hunspell'))
elif os.path.basename(current_dir) == 'Coderingstool':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, 'hunspell'))

# Hunspell settings
SUPPORTED_LANGUAGES = ["nl", "en_GB"]  # Dutch and English
HUNSPELL_PATH = os.path.join(hunspell_dir, "hunspell.exe")
DUTCH_DICT_PATH = os.path.join(hunspell_dir, "dict", "nl_NL")
ENGLISH_DICT_PATH = os.path.join(hunspell_dir, "dict", "en_GB")
DEFAULT_LANGUAGE = "Dutch"

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

def get_default_cache_dir():
    """Get the default cache directory relative to project root"""
    # Get the directory where this file is located (src)
    src_dir = Path(__file__).parent
    # Go up one level to project root, then into data/cache
    return src_dir.parent / "data" / "cache"


@dataclass
class CacheConfig:
    """Configuration for cache management system"""
    
    # Base cache directory - absolute path to project_root/data/cache
    cache_dir: Path = field(default_factory=get_default_cache_dir)
    
    # SQLite database name
    db_name: str = "cache.db"
    
    # Step prefixes for file naming
    step_prefixes: Dict[str, str] = field(default_factory=lambda: {
        "data": "001",
        "preprocessed": "002", 
        "segmented_descriptions": "003",
        "embeddings": "004",
        "clusters": "005",
        "labels": "006",
        "results": "007"
    })
    
    # Cache validity settings
    max_cache_age_days: int = 30
    check_file_hash: bool = True
    
    # File handling settings
    enable_compression: bool = False
    compression_level: int = 6  # 1-9, higher = more compression
    use_atomic_writes: bool = True
    
    # Performance settings
    batch_size: int = 1000
    memory_limit_mb: int = 500
    
    # Cleanup settings
    auto_cleanup: bool = True
    cleanup_interval_days: int = 7
    max_cache_size_gb: float = 10.0
    
    # Logging settings
    log_cache_operations: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Ensure cache directory exists and adjust settings for platform"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Disable atomic writes on Windows to avoid file locking issues
        import platform
        if platform.system() == 'Windows':
            self.use_atomic_writes = False
        
    @property
    def db_path(self) -> Path:
        """Full path to the SQLite database"""
        return self.cache_dir / self.db_name
    
    def get_step_prefix(self, step_name: str) -> str:
        """Get the numeric prefix for a given step"""
        return self.step_prefixes.get(step_name, "999")
    
    def get_cache_filename(self, original_filename: str, step_name: str) -> str:
        """Generate cache filename with prefix"""
        base_name = Path(original_filename).stem
        prefix = self.get_step_prefix(step_name)
        return f"{prefix}_{step_name}_{base_name}.csv"
    
    def get_cache_filepath(self, original_filename: str, step_name: str) -> Path:
        """Get full path for cached file"""
        cache_filename = self.get_cache_filename(original_filename, step_name)
        return self.cache_dir / cache_filename


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters that affect cache validity"""
    
    # Language settings
    language: str = "nl"
    spell_check_enabled: bool = True
    
    # Quality filter settings
    quality_threshold: float = 0.5
    min_response_length: int = 5
    
    # Clustering settings
    clustering_algorithm: str = "hdbscan"
    min_cluster_size: int = 5
    min_samples: int = 3
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    
    # Labeling settings
    labeling_model: str = "gpt-4o-mini"
    labeling_temperature: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def get_hash(self) -> str:
        """Generate hash of configuration for cache invalidation"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

@dataclass
class ClusteringConfig:
    """Configuration for automatic clustering mode."""
    
    # User-configurable options
    embedding_type: str = "description"  # Default to description as requested
    language: str = "nl"  # Default to Dutch
    
    # Quality thresholds for automatic decisions
    min_quality_score: float = 0.3  # Minimum acceptable quality
    max_noise_ratio: float = 0.5    # Maximum acceptable noise before micro-clustering
    
    # Optional overrides (mostly for testing)
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    
    def get_reducer_params(self, data_size: int) -> Dict:
        """Get UMAP parameters based on data size."""
        n_neighbors = min(30, max(15, data_size // 50))
        min_dist = 0.0 if data_size < 1000 else 0.1
        
        return {
            'n_neighbors': n_neighbors,
            'n_components': 10,
            'min_dist': min_dist,
            'metric': 'cosine',
            'random_state': 42,
            'n_jobs': 1
        }
    
    def get_auto_params(self, data_size: int) -> Dict:
        """Get clustering parameters based on data size."""
        # Use override values if provided
        if self.min_cluster_size is not None and self.min_samples is not None:
            return {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'
            }
        
        # Auto-calculate based on data size
        if data_size < 100:
            min_cluster_size = 2
            min_samples = 1
        elif data_size < 500:
            min_cluster_size = 3
            min_samples = 2
        elif data_size < 1000:
            min_cluster_size = 5
            min_samples = 3
        else:
            min_cluster_size = 10
            min_samples = 5
            
        return {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        }


@dataclass
class LabellerConfig:
    """Configuration for hierarchical labeller"""
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 4000
    language: str = DEFAULT_LANGUAGE
    max_retries: int = 3
    batch_size: int = 8  # Micro-clusters per batch
    top_k_representatives: int = 3  # Representative codes per cluster
    concurrent_requests: int = 5


# =============================================================================
# CACHE DATABASE
# =============================================================================

logger = logging.getLogger(__name__)


class CacheDatabase:
    """Manages SQLite database for cache metadata tracking"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.db_path = config.db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database tables if they don't exist"""
        with self._get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    cache_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    file_size INTEGER,
                    processing_time FLOAT,
                    parameters TEXT,
                    config_hash TEXT,
                    status TEXT DEFAULT 'valid',
                    UNIQUE(filename, step_name)
                );
                
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    success BOOLEAN,
                    error_message TEXT,
                    input_hash TEXT,
                    output_hash TEXT
                );
                
                CREATE TABLE IF NOT EXISTS clustering_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_quality REAL,
                    silhouette_score REAL,
                    noise_ratio REAL,
                    coverage REAL,
                    mean_cluster_size REAL,
                    size_variance REAL,
                    num_clusters INTEGER,
                    num_meta_clusters INTEGER,
                    embedding_type TEXT,
                    language TEXT,
                    min_quality_score REAL,
                    max_noise_ratio REAL,
                    parameters TEXT,
                    attempts INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_cache_filename_step 
                ON cache_metadata(filename, step_name);
                
                CREATE INDEX IF NOT EXISTS idx_cache_status 
                ON cache_metadata(status);
                
                CREATE INDEX IF NOT EXISTS idx_history_filename 
                ON processing_history(filename);
                
                CREATE INDEX IF NOT EXISTS idx_metrics_filename 
                ON clustering_metrics(filename);
            ''')
    
    def record_cache_entry(self, 
                          filename: str, 
                          step_name: str,
                          cache_path: str,
                          file_hash: str,
                          file_size: int,
                          processing_time: float = None,
                          parameters: Dict = None,
                          config_hash: str = None) -> int:
        """Record a new cache entry or update existing one"""
        with self._get_connection() as conn:
            params_json = json.dumps(parameters) if parameters else None
            
            # Insert or replace the cache entry
            cursor = conn.execute('''
                INSERT OR REPLACE INTO cache_metadata 
                (filename, step_name, cache_path, file_hash, file_size, 
                 processing_time, parameters, config_hash, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (filename, step_name, cache_path, file_hash, file_size, 
                  processing_time, params_json, config_hash))
            
            return cursor.lastrowid
    
    def get_cache_info(self, filename: str, step_name: str) -> Optional[Dict]:
        """Get cache metadata for a specific file and step"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM cache_metadata 
                WHERE filename = ? AND step_name = ? AND status = 'valid'
            ''', (filename, step_name))
            
            row = cursor.fetchone()
            if row:
                # Convert row to dictionary
                result = dict(row)
                # Parse JSON fields
                if result['parameters']:
                    result['parameters'] = json.loads(result['parameters'])
                # Update last accessed time
                conn.execute('''
                    UPDATE cache_metadata 
                    SET last_accessed = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (result['id'],))
                return result
            return None
    
    def is_cache_valid(self, 
                      filename: str, 
                      step_name: str,
                      max_age_days: Optional[int] = None,
                      config_hash: Optional[str] = None) -> bool:
        """Check if cache entry is valid based on age and configuration"""
        cache_info = self.get_cache_info(filename, step_name)
        
        if not cache_info:
            return False
        
        # Check if cache file exists
        cache_path = Path(cache_info['cache_path'])
        if not cache_path.exists():
            self.invalidate_cache(filename, step_name)
            return False
        
        # Check age
        if max_age_days is None:
            max_age_days = self.config.max_cache_age_days
        
        created_at = datetime.fromisoformat(cache_info['created_at'])
        age = datetime.now() - created_at
        
        if age > timedelta(days=max_age_days):
            return False
        
        # Check configuration hash if provided
        if config_hash and cache_info.get('config_hash') != config_hash:
            return False
        
        return True
    
    def invalidate_cache(self, 
                        filename: Optional[str] = None, 
                        step_name: Optional[str] = None):
        """Mark cache entries as invalid"""
        with self._get_connection() as conn:
            if filename and step_name:
                # Invalidate specific entry
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE filename = ? AND step_name = ?
                ''', (filename, step_name))
            elif filename:
                # Invalidate all steps for a file
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE filename = ?
                ''', (filename,))
            elif step_name:
                # Invalidate all files for a step
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE step_name = ?
                ''', (step_name,))
            else:
                # Invalidate everything
                conn.execute("UPDATE cache_metadata SET status = 'invalid'")
    
    def record_processing(self,
                         filename: str,
                         step_name: str,
                         started_at: datetime,
                         completed_at: datetime = None,
                         success: bool = True,
                         error_message: str = None,
                         input_hash: str = None,
                         output_hash: str = None):
        """Record processing history"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO processing_history 
                (filename, step_name, started_at, completed_at, 
                 success, error_message, input_hash, output_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, step_name, started_at, completed_at,
                  success, error_message, input_hash, output_hash))
    
    def get_processing_history(self, 
                              filename: str = None, 
                              step_name: str = None,
                              limit: int = 100) -> List[Dict]:
        """Get processing history"""
        with self._get_connection() as conn:
            query = "SELECT * FROM processing_history WHERE 1=1"
            params = []
            
            if filename:
                query += " AND filename = ?"
                params.append(filename)
            
            if step_name:
                query += " AND step_name = ?"
                params.append(step_name)
            
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_entries(self, days_to_keep: int = None):
        """Remove old cache entries and their files"""
        if days_to_keep is None:
            days_to_keep = self.config.max_cache_age_days
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self._get_connection() as conn:
            # Get files to delete
            cursor = conn.execute('''
                SELECT cache_path FROM cache_metadata 
                WHERE created_at < ? OR status = 'invalid'
            ''', (cutoff_date,))
            
            files_to_delete = [row['cache_path'] for row in cursor.fetchall()]
            
            # Delete database entries
            conn.execute('''
                DELETE FROM cache_metadata 
                WHERE created_at < ? OR status = 'invalid'
            ''', (cutoff_date,))
            
            # Delete old processing history
            conn.execute('''
                DELETE FROM processing_history 
                WHERE started_at < ?
            ''', (cutoff_date,))
        
        # Delete actual files
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                Path(file_path).unlink(missing_ok=True)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
        
        return deleted_count
    
    def get_cache_statistics(self) -> Dict:
        """Get cache usage statistics"""
        with self._get_connection() as conn:
            stats = {}
            
            # Total cache entries
            cursor = conn.execute("SELECT COUNT(*) as count FROM cache_metadata WHERE status = 'valid'")
            stats['total_entries'] = cursor.fetchone()['count']
            
            # Cache size
            cursor = conn.execute("SELECT SUM(file_size) as total_size FROM cache_metadata WHERE status = 'valid'")
            stats['total_size_bytes'] = cursor.fetchone()['total_size'] or 0
            
            # Entries by step
            cursor = conn.execute('''
                SELECT step_name, COUNT(*) as count, SUM(file_size) as size 
                FROM cache_metadata 
                WHERE status = 'valid' 
                GROUP BY step_name
            ''')
            stats['by_step'] = {row['step_name']: {
                'count': row['count'], 
                'size': row['size'] or 0
            } for row in cursor.fetchall()}
            
            # Recent processing
            cursor = conn.execute('''
                SELECT COUNT(*) as count, 
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
                FROM processing_history 
                WHERE started_at > datetime('now', '-7 days')
            ''')
            row = cursor.fetchone()
            stats['recent_processing'] = {
                'total': row['count'],
                'successful': row['success_count']
            }
            
            return stats
    
    def record_clustering_metrics(self, 
                                filename: str, 
                                metrics: Dict,
                                config: ClusteringConfig,
                                attempts: int = 1) -> int:
        """Record clustering quality metrics"""
        with self._get_connection() as conn:
            # Extract metrics
            overall_quality = metrics.get('overall_quality', 0)
            silhouette_score = metrics.get('silhouette_score', 0)
            noise_ratio = metrics.get('noise_ratio', 0)
            coverage = metrics.get('coverage', 0)
            mean_cluster_size = metrics.get('mean_cluster_size', 0)
            size_variance = metrics.get('size_variance', 0)
            num_clusters = metrics.get('num_clusters', 0)
            num_meta_clusters = metrics.get('num_meta_clusters', 0)
            
            # Get additional metrics from config
            embedding_type = config.embedding_type
            language = config.language
            min_quality_score = config.min_quality_score
            max_noise_ratio = config.max_noise_ratio
            
            # Extract parameters used (min_samples, min_cluster_size)
            parameters = json.dumps({
                'min_samples': config.min_samples if hasattr(config, 'min_samples') else None,
                'min_cluster_size': config.min_cluster_size if hasattr(config, 'min_cluster_size') else None
            })
            
            cursor = conn.execute('''
                INSERT INTO clustering_metrics 
                (filename, overall_quality, silhouette_score, noise_ratio, coverage,
                 mean_cluster_size, size_variance, num_clusters, num_meta_clusters,
                 embedding_type, language, min_quality_score, max_noise_ratio, 
                 parameters, attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, overall_quality, silhouette_score, noise_ratio, coverage,
                  mean_cluster_size, size_variance, num_clusters, num_meta_clusters,
                  embedding_type, language, min_quality_score, max_noise_ratio, 
                  parameters, attempts))
            
            return cursor.lastrowid
    
    def get_clustering_metrics(self, filename: str, limit: int = 1) -> List[Dict]:
        """Get clustering metrics for a file"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM clustering_metrics 
                WHERE filename = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (filename, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result['parameters']:
                    result['parameters'] = json.loads(result['parameters'])
                results.append(result)
            
            return results


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Global configuration instances
DEFAULT_CACHE_CONFIG = CacheConfig()
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()
DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()
DEFAULT_LABELLER_CONFIG = LabellerConfig()

# Environment-based overrides
if os.getenv("CODERINGSTOOL_CACHE_DIR"):
    DEFAULT_CACHE_CONFIG.cache_dir = Path(os.getenv("CODERINGSTOOL_CACHE_DIR"))

if os.getenv("CODERINGSTOOL_MAX_CACHE_AGE"):
    DEFAULT_CACHE_CONFIG.max_cache_age_days = int(os.getenv("CODERINGSTOOL_MAX_CACHE_AGE"))

if os.getenv("CODERINGSTOOL_VERBOSE"):
    DEFAULT_CACHE_CONFIG.verbose = True