import csv
import hashlib
import json
import logging
import os
import pickle
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Type, TypeVar, List, Optional, Dict

import numpy as np
from pydantic import BaseModel

from config import CacheConfig, ProcessingConfig, ClusteringConfig

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


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
                result = dict(row)
                if result['parameters']:
                    result['parameters'] = json.loads(result['parameters'])
                
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
        
        cache_path = Path(cache_info['cache_path'])
        if not cache_path.exists():
            self.invalidate_cache(filename, step_name)
            return False
        
        if max_age_days is None:
            max_age_days = self.config.max_cache_age_days
        
        created_at = datetime.fromisoformat(cache_info['created_at'])
        age = datetime.now() - created_at
        
        if age > timedelta(days=max_age_days):
            return False
        
        if config_hash and cache_info.get('config_hash') != config_hash:
            return False
        
        return True
    
    def invalidate_cache(self, 
                        filename: Optional[str] = None, 
                        step_name: Optional[str] = None):
        """Mark cache entries as invalid"""
        with self._get_connection() as conn:
            if filename and step_name:
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
            elif step_name:
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE step_name = ?
                ''', (step_name,))
            else:
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
            cursor = conn.execute('''
                SELECT cache_path FROM cache_metadata 
                WHERE created_at < ? OR status = 'invalid'
            ''', (cutoff_date,))
            
            files_to_delete = [row['cache_path'] for row in cursor.fetchall()]
            
            conn.execute('''
                DELETE FROM cache_metadata 
                WHERE created_at < ? OR status = 'invalid'
            ''', (cutoff_date,))
            
            conn.execute('''
                DELETE FROM processing_history 
                WHERE started_at < ?
            ''', (cutoff_date,))
        
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
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM cache_metadata WHERE status = 'valid'")
            stats['total_entries'] = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT SUM(file_size) as total_size FROM cache_metadata WHERE status = 'valid'")
            stats['total_size_bytes'] = cursor.fetchone()['total_size'] or 0
            
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
            overall_quality = metrics.get('overall_quality', 0)
            silhouette_score = metrics.get('silhouette_score', 0)
            noise_ratio = metrics.get('noise_ratio', 0)
            coverage = metrics.get('coverage', 0)
            mean_cluster_size = metrics.get('mean_cluster_size', 0)
            size_variance = metrics.get('size_variance', 0)
            num_clusters = metrics.get('num_clusters', 0)
            num_meta_clusters = metrics.get('num_meta_clusters', 0)
            
            embedding_type = config.embedding_type
            language = config.language
            min_quality_score = config.min_quality_score
            max_noise_ratio = config.max_noise_ratio
            
            parameters = json.dumps({
                'min_samples': getattr(config, 'min_samples', None),
                'min_cluster_size': getattr(config, 'min_cluster_size', None)
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


class CacheManager:
    """Manages cached data with SQLite metadata tracking"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.db = CacheDatabase(self.config)
        
        # Set CSV field size limit for large fields
        csv.field_size_limit(2**31 - 1)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_array_field(self, value) -> bool:
        """Check if a value should be treated as an array"""
        return (isinstance(value, list) and 
                len(value) > 0 and 
                all(isinstance(x, (int, float)) for x in value))
    
    def _serialize_value(self, obj):
        """Recursively serialize complex objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, list):
            return [self._serialize_value(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        else:
            return obj
    
    def _convert_numpy_scalar(self, value):
        """Convert numpy scalar types to Python types"""
        if isinstance(value, (np.integer, np.int_)):
            return int(value)
        elif isinstance(value, (np.floating, np.float_)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return value
    
    def _deserialize_value(self, value, field_name: str = None):
        """Deserialize value from CSV, converting to numpy arrays where appropriate"""
        if value == '':
            return None
        
        # Try to parse as JSON for complex types
        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
            try:
                parsed = json.loads(value)
                
                # Convert lists of numbers to numpy arrays for embedding fields
                if field_name and 'embedding' in field_name and self._is_array_field(parsed):
                    return np.array(parsed, dtype=np.float32)
                
                # Handle nested structures
                if isinstance(parsed, dict):
                    for key, val in parsed.items():
                        if 'embedding' in key and self._is_array_field(val):
                            parsed[key] = np.array(val, dtype=np.float32)
                elif isinstance(parsed, list):
                    # Handle list of response segments
                    for item in parsed:
                        if isinstance(item, dict):
                            for key, val in item.items():
                                if 'embedding' in key and self._is_array_field(val):
                                    item[key] = np.array(val, dtype=np.float32)
                
                return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Handle primitive types
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            # Try numeric conversion
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
    
    def get_cache_path(self, filename: str, step: str) -> Path:
        """Get the cache file path for a given step"""
        return self.config.get_cache_filepath(filename, step)
    
    def is_cache_valid(self, 
                      filename: str, 
                      step: str,
                      processing_config: ProcessingConfig = None) -> bool:
        """Check if cached data exists and is valid"""
        config_hash = processing_config.get_hash() if processing_config else None
        return self.db.is_cache_valid(filename, step, config_hash=config_hash)
    
    def load_from_cache(self, 
                       filename: str, 
                       step: str, 
                       model_cls: Type[T]) -> Optional[List[T]]:
        """Load data from cache if valid"""
        cache_info = self.db.get_cache_info(filename, step)
        
        if not cache_info:
            logger.info(f"No cache found for {filename} at step {step}")
            return None
        
        cache_path = Path(cache_info['cache_path'])
        
        if not cache_path.exists():
            logger.warning(f"Cache file missing: {cache_path}")
            self.db.invalidate_cache(filename, step)
            return None
        
        try:
            result = []
            with open(cache_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    parsed_row = {}
                    for key, value in row.items():
                        parsed_row[key] = self._deserialize_value(value, key)
                    
                    # Create model instance
                    result.append(model_cls(**parsed_row))
            
            logger.info(f"Loaded {len(result)} items from cache for {filename} at step {step}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading cache for {filename} at step {step}: {e}")
            self.db.invalidate_cache(filename, step)
            return None
    
    def _write_csv_file(self, filepath: Path, data: List[T]):
        """Write data to CSV file"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Get field names from first item
            sample = data[0]
            serialize_method = getattr(sample, 'model_dump', None) or getattr(sample, 'dict')
            fieldnames = list(serialize_method().keys())
            writer.writerow(fieldnames)
            
            # Write data rows
            for item in data:
                row = []
                for field in fieldnames:
                    value = getattr(item, field)
                    
                    if value is None:
                        row.append('')
                    elif isinstance(value, (list, dict)) or hasattr(value, 'model_dump') or hasattr(value, 'dict'):
                        row.append(json.dumps(self._serialize_value(value), cls=NumpyEncoder))
                    else:
                        row.append(self._convert_numpy_scalar(value))
                
                writer.writerow(row)
    
    def save_to_cache(self, 
                     data: List[T], 
                     filename: str, 
                     step: str,
                     processing_config: ProcessingConfig = None,
                     processing_time: float = None) -> bool:
        """Save data to cache with atomic write"""
        if not data:
            logger.warning(f"No data to save for {filename} at step {step}")
            return False
        
        cache_path = self.get_cache_path(filename, step)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use atomic writes with temporary file if enabled
            if self.config.use_atomic_writes:
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=cache_path.parent, 
                    prefix=f".{cache_path.stem}_", 
                    suffix='.tmp'
                )
                write_path = Path(temp_path)
                os.close(temp_fd)  # Close the file descriptor immediately
            else:
                write_path = cache_path
            
            # Write the CSV file
            self._write_csv_file(write_path, data)
            
            # Move to final location if using atomic writes
            if self.config.use_atomic_writes:
                try:
                    write_path.replace(cache_path)
                except (PermissionError, OSError):
                    # Fallback for Windows
                    shutil.move(str(write_path), str(cache_path))
            
            # Record in database
            file_hash = self._calculate_file_hash(cache_path)
            file_size = cache_path.stat().st_size
            config_hash = processing_config.get_hash() if processing_config else None
            
            self.db.record_cache_entry(
                filename=filename,
                step_name=step,
                cache_path=str(cache_path),
                file_hash=file_hash,
                file_size=file_size,
                processing_time=processing_time,
                config_hash=config_hash
            )
            
            logger.info(f"Saved {len(data)} items to cache for {filename} at step {step}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cache for {filename} at step {step}: {e}")
            # Clean up temporary file if it exists
            if self.config.use_atomic_writes and 'write_path' in locals() and write_path.exists():
                try:
                    write_path.unlink()
                except:
                    pass
            return False
    
    def invalidate_cache(self, filename: str = None, step: str = None):
        """Invalidate cache entries"""
        self.db.invalidate_cache(filename, step)
    
    def cleanup_old_cache(self):
        """Remove old cache entries and files"""
        if self.config.auto_cleanup:
            deleted_count = self.db.cleanup_old_entries()
            logger.info(f"Cleaned up {deleted_count} old cache files")
    
    def get_statistics(self) -> Dict:
        """Get cache usage statistics"""
        return self.db.get_cache_statistics()
    
    def cache_intermediate_data(self, data, filename: str, cache_key: str):
        """Cache intermediate processing data for phase-to-phase communication"""
        cache_dir = self.config.cache_dir / "intermediate"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_path = cache_dir / f"{filename}_{cache_key}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cached intermediate data to {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error caching intermediate data: {e}")
            return False
    
    def load_intermediate_data(self, filename: str, cache_key: str, expected_type=None):
        """Load intermediate processing data"""
        cache_dir = self.config.cache_dir / "intermediate"
        cache_path = cache_dir / f"{filename}_{cache_key}.pkl"
        
        if not cache_path.exists():
            logger.warning(f"No cached intermediate data found at {cache_path}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded intermediate data from {cache_path}")
            
            # Optional type checking
            if expected_type and not isinstance(data, expected_type):
                logger.warning(f"Loaded data is not of expected type {expected_type}")
                return None
            
            return data
        except Exception as e:
            logger.error(f"Error loading intermediate data: {e}")
            return None
    
    def save_clustering_metrics(self, filename: str, metrics: Dict, config: ClusteringConfig):
        """Save clustering quality metrics to database"""
        # Get number of attempts from metrics if available
        attempts = len(metrics) if isinstance(metrics, list) else 1
        
        # If metrics is a list, save the final metrics
        final_metrics = metrics[-1] if isinstance(metrics, list) else metrics
        
        try:
            # Update config with actual parameters if available in metrics
            if 'parameters' in final_metrics:
                params = final_metrics['parameters']
                if 'min_samples' in params:
                    config.min_samples = params['min_samples']
                if 'min_cluster_size' in params:
                    config.min_cluster_size = params['min_cluster_size']
            
            # Record metrics in database
            self.db.record_clustering_metrics(
                filename=filename,
                metrics=final_metrics,
                config=config,
                attempts=attempts
            )
            logger.info(f"Saved clustering metrics for {filename}")
            
        except Exception as e:
            logger.error(f"Error saving clustering metrics: {e}")
    
    def get_clustering_metrics(self, filename: str, limit: int = 1) -> List[Dict]:
        """Get clustering metrics for a file"""
        return self.db.get_clustering_metrics(filename, limit)