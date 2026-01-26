import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import pickle
import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Type, TypeVar, List, Optional, Dict

# === MODELS ========================================================================================================
from pydantic import BaseModel

# === CONFIG ========================================================================================================
from config import CacheConfig

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


def generate_variable_key(selected_variables: List[str], is_merged: bool = False) -> str:
    """
    Generate a cache key based on selected variables
    
    Args:
        selected_variables: List of variable names (e.g., ['Q18'] or ['Q18', 'Q19', 'Q20'])
        is_merged: Whether this represents merged variables
    
    Returns:
        str: Variable key for caching (e.g., 'Q18' or 'Q18+Q19+Q20')
    """
    if not selected_variables:
        return "unknown"
    
    # For single variable or if not merged, return first variable
    if not is_merged or len(selected_variables) == 1:
        return selected_variables[0]
    
    # For multiple merged variables, sort for consistency and join with +
    # This ensures Q18+Q19 is the same as Q19+Q18
    sorted_vars = sorted(selected_variables)
    return "+".join(sorted_vars)


def generate_enhanced_variable_key(selected_variables: List[str],
                                  is_merged: bool = False, sample_size: Optional[int] = None,
                                  merge_config: Optional[dict] = None) -> str:
    """
    Generate enhanced variable key including sample size and merge configuration for cache operations

    Args:
        selected_variables: List of variable names (e.g., ['Q18'] or ['Q18', 'Q19', 'Q20'])
        is_merged: Whether this represents merged variables
        sample_size: Sample size for truncation (None means no truncation)
        merge_config: Merge configuration dict with 'separator', 'strategy', 'skip_empty' (for merged variables)

    Returns:
        str: Enhanced variable key (e.g., 'Q18_full' or 'Q18+Q19_concat_semicolon_skip_250')
    """
    # Generate base variable key
    base_key = generate_variable_key(selected_variables, is_merged)

    # Add merge configuration for merged variables
    merge_suffix = ""
    if is_merged and merge_config:
        strategy = merge_config.get('strategy', 'concatenate')
        separator = merge_config.get('separator', ' ')
        skip_empty = merge_config.get('skip_empty', True)

        # Map strategy to short code
        strategy_code = {
            'concatenate': 'concat',
            'first_available': 'first',
            'prioritized': 'prior',
            'all_combined': 'allcomb'
        }.get(strategy, strategy[:6])

        # Map separator to short code
        sep_code = {
            ' ': 'space',
            '\n': 'newline',
            '; ': 'semicolon',
            ', ': 'comma',
            ' | ': 'pipe'
        }.get(separator, 'custom')

        # Build merge suffix
        skip_code = 'skip' if skip_empty else 'noskip'
        merge_suffix = f"_{strategy_code}_{sep_code}_{skip_code}"

    # Add sample size suffix
    sample_suffix = f"_{sample_size}" if sample_size else "_full"

    return f"{base_key}{merge_suffix}{sample_suffix}"


def generate_enhanced_cache_key(filename: str, selected_variables: List[str],
                               is_merged: bool = False, sample_size: Optional[int] = None,
                               merge_config: Optional[dict] = None) -> str:
    """
    Generate enhanced cache key including sample size and merge configuration for consistent caching

    Args:
        filename: SPSS filename (e.g., 'survey.sav')
        selected_variables: List of variable names (e.g., ['Q18'] or ['Q18', 'Q19', 'Q20'])
        is_merged: Whether this represents merged variables
        sample_size: Sample size for truncation (None means no truncation)
        merge_config: Merge configuration dict (for merged variables)

    Returns:
        str: Enhanced cache key (e.g., 'survey_Q18_full' or 'survey_Q18+Q19_concat_semicolon_skip_250')
    """
    # Get base filename without extension
    base_filename = filename.replace('.sav', '')

    # Generate enhanced variable key
    variable_key = generate_enhanced_variable_key(selected_variables, is_merged, sample_size, merge_config)

    return f"{base_filename}_{variable_key}"


class CacheDatabase:
    """Simple SQLite database for cache metadata tracking"""
    
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
                    variable_key TEXT NOT NULL,
                    cache_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    file_size INTEGER,
                    processing_time FLOAT,
                    status TEXT DEFAULT 'valid',
                    var_lab TEXT,
                    UNIQUE(filename, step_name, variable_key)
                );

                CREATE INDEX IF NOT EXISTS idx_cache_filename_step_var
                ON cache_metadata(filename, step_name, variable_key);

                CREATE INDEX IF NOT EXISTS idx_cache_status
                ON cache_metadata(status);
            ''')

            # Migration: Add var_lab column to existing databases
            try:
                conn.execute('ALTER TABLE cache_metadata ADD COLUMN var_lab TEXT;')
                logger.info("Added var_lab column to existing cache_metadata table")
            except sqlite3.OperationalError as e:
                # Column already exists - this is fine
                if "duplicate column name" in str(e).lower():
                    pass
                else:
                    raise
    
    def record_cache_entry(self,
                          filename: str,
                          step_name: str,
                          variable_key: str,
                          cache_path: str,
                          file_hash: str,
                          file_size: int,
                          processing_time: float = None,
                          var_lab: str = None) -> int:
        """Record a new cache entry or update existing one"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                INSERT OR REPLACE INTO cache_metadata
                (filename, step_name, variable_key, cache_path, file_hash, file_size,
                 processing_time, var_lab, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (filename, step_name, variable_key, cache_path, file_hash, file_size, processing_time, var_lab))

            return cursor.lastrowid
    
    def get_cache_info(self, filename: str, step_name: str, variable_key: str) -> Optional[Dict]:
        """Get cache metadata for a specific file, step, and variable"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM cache_metadata 
                WHERE filename = ? AND step_name = ? AND variable_key = ? AND status = 'valid'
            ''', (filename, step_name, variable_key))
            
            row = cursor.fetchone()
            if row:
                # Convert row to dictionary and update last accessed time
                result = dict(row)
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
                      variable_key: str,
                      max_age_days: Optional[int] = None) -> bool:
        """Check if cache entry is valid based on age"""
        cache_info = self.get_cache_info(filename, step_name, variable_key)
        
        if not cache_info:
            return False
        
        # Check if cache file exists
        cache_path = Path(cache_info['cache_path'])
        if not cache_path.exists():
            self.invalidate_cache(filename, step_name, variable_key)
            return False
        
        # Check age
        if max_age_days is None:
            max_age_days = self.config.max_cache_age_days
        
        created_at = datetime.fromisoformat(cache_info['created_at'])
        age = datetime.now() - created_at
        
        return age <= timedelta(days=max_age_days)

    def get_all_cached_steps(self, filename: str, variable_key: str) -> List[str]:
        """Get all valid cached step names for a filename and variable_key"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT DISTINCT step_name FROM cache_metadata
                WHERE filename = ? AND variable_key = ? AND status = 'valid'
                ORDER BY step_name
            ''', (filename, variable_key))
            return [row['step_name'] for row in cursor.fetchall()]

    def invalidate_cache(self,
                        filename: Optional[str] = None, 
                        step_name: Optional[str] = None,
                        variable_key: Optional[str] = None):
        """Mark cache entries as invalid"""
        with self._get_connection() as conn:
            if filename and step_name and variable_key:
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE filename = ? AND step_name = ? AND variable_key = ?
                ''', (filename, step_name, variable_key))
            elif filename and step_name:
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE filename = ? AND step_name = ?
                ''', (filename, step_name))
            elif filename:
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE filename = ?
                ''', (filename,))
            else:
                conn.execute("UPDATE cache_metadata SET status = 'invalid'")


class CacheManager:
    """Simple, robust cache manager using pickle storage for Pydantic models"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.db = CacheDatabase(self.config)

        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Automatically cleanup old cache if enabled
        if self.config.auto_cleanup:
            self.cleanup_old_cache()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file with retry for Windows file handle issues"""
        import gc
        import time

        for attempt in range(3):  # Try up to 3 times
            try:
                hash_md5 = hashlib.md5()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()
            except Exception as e:
                if "closed file" in str(e).lower() and attempt < 2:
                    # Windows file handle not released yet, retry with backoff
                    gc.collect()
                    time.sleep(0.05 * (attempt + 1))  # Increasing backoff: 50ms, 100ms
                    continue
                raise
    
    def get_cache_path(self, filename: str, step: str, variable_key: str) -> Path:
        """Get the cache file path for a given step and variable"""
        base_name = Path(filename).stem
        prefix = self.config.get_step_prefix(step)
        cache_filename = f"{prefix}_{step}_{base_name}_{variable_key}.pkl"
        return self.config.cache_dir / cache_filename
    
    def is_cache_valid(self, filename: str, step: str, variable_key: str) -> bool:
        """Check if cached data exists and is valid"""
        return self.db.is_cache_valid(filename, step, variable_key)
    
    def save_to_cache(self, data: List[T], filename: str, step: str, variable_key: str, processing_time: float = None, var_lab: str = None) -> bool:
        """Save list of Pydantic models to cache using pickle"""
        if not data:
            logger.warning(f"No data to save for {filename} at step {step} with variable {variable_key}")
            return False

        cache_path = self.get_cache_path(filename, step, variable_key)

        try:
            # Convert Pydantic models to dictionaries for serialization
            serializable_data = [item.model_dump() for item in data]

            # Save using pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(serializable_data, f)
                f.flush()
                os.fsync(f.fileno())  # Force Windows to flush file handle

            # Calculate file hash and size
            file_hash = self._calculate_file_hash(cache_path)
            file_size = cache_path.stat().st_size

            # Record in database
            self.db.record_cache_entry(
                filename=filename,
                step_name=step,
                variable_key=variable_key,
                cache_path=str(cache_path),
                file_hash=file_hash,
                file_size=file_size,
                processing_time=processing_time,
                var_lab=var_lab
            )

            logger.info(f"Saved {len(data)} items to cache for {filename} at step {step} with variable {variable_key}")
            return True

        except Exception as e:
            logger.error(f"Error saving cache for {filename} at step {step} with variable {variable_key}: {e}")
            # Clean up partial file if it exists
            if cache_path.exists():
                cache_path.unlink()
            return False
    
    def _safe_pickle_load(self, path: Path):
        """
        Safely load pickle file with retry logic to handle file handle issues on Windows.

        This addresses the "I/O operation on closed file" error that can occur when
        pickle.load() is called on a file that hasn't been fully released by the OS.

        Args:
            path: Path to the pickle file

        Returns:
            Unpickled data

        Raises:
            Exception: If loading fails after retry
        """
        import gc
        import time

        for attempt in range(2):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except ValueError as e:
                # Typical message: "I/O operation on closed file"
                if "closed file" in str(e) and attempt == 0:
                    logger.warning(f"File handle issue on attempt {attempt + 1}, retrying after gc and sleep...")
                    gc.collect()
                    time.sleep(0.05)
                    continue
                raise
            except Exception as e:
                # Handle other exceptions on first attempt
                if attempt == 0 and "closed file" in str(e).lower():
                    logger.warning(f"File handle issue on attempt {attempt + 1}, retrying after gc and sleep...")
                    gc.collect()
                    time.sleep(0.05)
                    continue
                raise

    def load_from_cache(self,  filename: str,  step: str,  variable_key: str, model_cls: Type[T]) -> Optional[List[T]]:
        """Load data from cache and reconstruct Pydantic models"""
        cache_info = self.db.get_cache_info(filename, step, variable_key)

        if not cache_info:
            logger.info(f"No cache found for {filename} at step {step} with variable {variable_key}")
            return None

        cache_path = Path(cache_info['cache_path'])

        if not cache_path.exists():
            logger.warning(f"Cache file missing: {cache_path}")
            self.db.invalidate_cache(filename, step, variable_key)
            return None

        try:
            # Load pickled data using safe loader with retry logic
            serializable_data = self._safe_pickle_load(cache_path)

            # Reconstruct Pydantic models
            result = [model_cls.model_validate(item_data) for item_data in serializable_data]

            logger.info(f"Loaded {len(result)} items from cache for {filename} at step {step} with variable {variable_key}")
            return result

        except ValueError as e:
            # Specific handling for "I/O operation on closed file" errors
            if "closed file" in str(e).lower():
                logger.error(f"File handle error loading cache for {filename} at step {step}: {e}")
                logger.error(f"Cache file path: {cache_path}")
                self.db.invalidate_cache(filename, step, variable_key)
                return None
            else:
                raise
        except Exception as e:
            logger.error(f"Error loading cache for {filename} at step {step} with variable {variable_key}: {e}")
            self.db.invalidate_cache(filename, step, variable_key)
            return None
    
    def invalidate_cache(self, filename: str = None, step: str = None, variable_key: str = None):
        """Invalidate cache entries"""
        self.db.invalidate_cache(filename, step, variable_key)
    
    def cleanup_old_cache(self) -> int:
        """Remove old cache entries and files"""
        if not self.config.auto_cleanup:
            return 0
        
        # Get cache info for files to delete
        with self.db._get_connection() as conn:
            cutoff_date = datetime.now() - timedelta(days=self.config.max_cache_age_days)
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
        
        # Delete actual files
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                Path(file_path).unlink(missing_ok=True)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old cache files")
        return deleted_count

    def get_cached_steps_for_dataset(self, filename: str, variable_key: str) -> list:
        """
        Return list of step numbers that have valid cache entries for a dataset

        Args:
            filename: The SPSS filename
            variable_key: The variable key (e.g., 'Q1_1+Q1_10+..._100')

        Returns:
            List of integers representing cached step numbers (e.g., [1, 2, 3])
        """
        # Mapping of step names to step numbers
        step_mapping = {
            "data": 1,
            "preprocessed": 2,
            "quality_filter": 3,
            "extracted_ideas": 4,
            "embeddings": 5,
            "initial_clusters": 6,
            "codebook_generation": 7,
            "theme_enriched_codebook": 8,
            "code_assignment_direct": 9,
            "code_assignment": 9,  # Alternative name for step 9
            "export": 10
        }

        cached_steps = []
        for step_name, step_num in step_mapping.items():
            if self.is_cache_valid(filename, step_name, variable_key):
                # Avoid duplicates (for alternative names like code_assignment)
                if step_num not in cached_steps:
                    cached_steps.append(step_num)

        return sorted(cached_steps)

    def get_statistics(self) -> Dict:
        """Get cache usage statistics"""
        with self.db._get_connection() as conn:
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

            return stats

    # =========================================================================
    # METADATA CACHING (single Pydantic model, not list)
    # =========================================================================

    def save_metadata_to_cache(
        self,
        metadata: T,
        filename: str,
        step: str,
        variable_key: str,
        processing_time: float = None,
        var_lab: str = None
    ) -> bool:
        """
        Save a single Pydantic model (metadata) to cache.

        Unlike save_to_cache() which works with lists, this method handles
        single metadata objects like ExtractionMetadata.

        Args:
            metadata: Single Pydantic BaseModel instance
            filename: SPSS filename
            step: Step name (will be stored with '_metadata' suffix)
            variable_key: Variable key for cache identification
            processing_time: Optional processing time in seconds
            var_lab: Optional variable label

        Returns:
            True if saved successfully, False otherwise
        """
        if metadata is None:
            logger.warning(f"No metadata to save for {filename} at step {step} with variable {variable_key}")
            return False

        # Use _metadata suffix to distinguish from list caches
        metadata_step = f"{step}_metadata"
        cache_path = self.get_cache_path(filename, metadata_step, variable_key)

        try:
            # Convert Pydantic model to dictionary for serialization
            serializable_data = metadata.model_dump()

            # Save using pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(serializable_data, f)
                f.flush()
                os.fsync(f.fileno())

            # Calculate file hash and size
            file_hash = self._calculate_file_hash(cache_path)
            file_size = cache_path.stat().st_size

            # Record in database
            self.db.record_cache_entry(
                filename=filename,
                step_name=metadata_step,
                variable_key=variable_key,
                cache_path=str(cache_path),
                file_hash=file_hash,
                file_size=file_size,
                processing_time=processing_time,
                var_lab=var_lab
            )

            logger.info(f"Saved metadata to cache for {filename} at step {metadata_step} with variable {variable_key}")
            return True

        except Exception as e:
            logger.error(f"Error saving metadata cache for {filename} at step {step} with variable {variable_key}: {e}")
            # Clean up partial file if it exists
            if cache_path.exists():
                cache_path.unlink()
            return False

    def load_metadata_from_cache(
        self,
        filename: str,
        step: str,
        variable_key: str,
        model_cls: Type[T]
    ) -> Optional[T]:
        """
        Load a single Pydantic model (metadata) from cache.

        Unlike load_from_cache() which returns a list, this method returns
        a single model instance.

        Args:
            filename: SPSS filename
            step: Step name (will look for '_metadata' suffix)
            variable_key: Variable key for cache identification
            model_cls: Pydantic model class to reconstruct

        Returns:
            Single Pydantic model instance, or None if not found/invalid
        """
        # Use _metadata suffix to match save_metadata_to_cache
        metadata_step = f"{step}_metadata"
        cache_info = self.db.get_cache_info(filename, metadata_step, variable_key)

        if not cache_info:
            logger.info(f"No metadata cache found for {filename} at step {metadata_step} with variable {variable_key}")
            return None

        cache_path = Path(cache_info['cache_path'])

        if not cache_path.exists():
            logger.warning(f"Metadata cache file missing: {cache_path}")
            self.db.invalidate_cache(filename, metadata_step, variable_key)
            return None

        try:
            # Load pickled data using safe loader with retry logic
            serializable_data = self._safe_pickle_load(cache_path)

            # Reconstruct single Pydantic model
            result = model_cls.model_validate(serializable_data)

            logger.info(f"Loaded metadata from cache for {filename} at step {metadata_step} with variable {variable_key}")
            return result

        except ValueError as e:
            if "closed file" in str(e).lower():
                logger.error(f"File handle error loading metadata cache for {filename} at step {metadata_step}: {e}")
                self.db.invalidate_cache(filename, metadata_step, variable_key)
                return None
            else:
                raise
        except Exception as e:
            logger.error(f"Error loading metadata cache for {filename} at step {metadata_step} with variable {variable_key}: {e}")
            self.db.invalidate_cache(filename, metadata_step, variable_key)
            return None

    def is_metadata_cache_valid(self, filename: str, step: str, variable_key: str) -> bool:
        """Check if metadata cache exists and is valid."""
        metadata_step = f"{step}_metadata"
        return self.db.is_cache_valid(filename, metadata_step, variable_key)


# # For backward compatibility, keep the cache_intermediate_data methods
# class CacheManager(CacheManager):
#     """Extended cache manager with intermediate data caching"""
    
#     def cache_intermediate_data(self, data, filename: str, cache_key: str) -> bool:
#         """Cache intermediate processing data for phase-to-phase communication"""
#         cache_dir = self.config.cache_dir / "intermediate"
#         cache_dir.mkdir(parents=True, exist_ok=True)
        
#         cache_path = cache_dir / f"{filename}_{cache_key}.pkl"
        
#         try:
#             with open(cache_path, 'wb') as f:
#                 pickle.dump(data, f)
#             logger.info(f"Cached intermediate data to {cache_path}")
#             return True
#         except Exception as e:
#             logger.error(f"Error caching intermediate data: {e}")
#             return False
    
#     def load_intermediate_data(self, filename: str, cache_key: str, expected_type=None):
#         """Load intermediate processing data"""
#         cache_dir = self.config.cache_dir / "intermediate"
#         cache_path = cache_dir / f"{filename}_{cache_key}.pkl"
        
#         if not cache_path.exists():
#             logger.warning(f"No cached intermediate data found at {cache_path}")
#             return None
        
#         try:
#             with open(cache_path, 'rb') as f:
#                 data = pickle.load(f)
#             logger.info(f"Loaded intermediate data from {cache_path}")
            
#             # Optional type checking
#             if expected_type and not isinstance(data, expected_type):
#                 logger.warning(f"Loaded data is not of expected type {expected_type}")
#                 return None
            
#             return data
#         except Exception as e:
#             logger.error(f"Error loading intermediate data: {e}")
#             return None