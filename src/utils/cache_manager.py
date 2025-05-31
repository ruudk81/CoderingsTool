import csv
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import shutil
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Type, TypeVar, List, Optional, Dict, get_type_hints, get_origin, get_args
from pydantic import BaseModel
import numpy as np

from ..config import CacheConfig, ProcessingConfig


logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


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
                
                CREATE INDEX IF NOT EXISTS idx_cache_filename_step 
                ON cache_metadata(filename, step_name);
                
                CREATE INDEX IF NOT EXISTS idx_cache_status 
                ON cache_metadata(status);
                
                CREATE INDEX IF NOT EXISTS idx_history_filename 
                ON processing_history(filename);
                
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
    
    def cleanup_old_entries(self, days_to_keep: int = None) -> List[str]:
        """Remove old cache entries from database and return list of files to delete"""
        if days_to_keep is None:
            days_to_keep = self.config.max_cache_age_days
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self._get_connection() as conn:
            # Get files to delete before removing DB entries
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
            
        
        return files_to_delete
    
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
    
    def _find_ndarray_fields(self, model_cls: Type[BaseModel], prefix: str = '') -> List[str]:
        """Recursively find all fields that are numpy ndarrays"""
        array_fields = []
        
        try:
            hints = get_type_hints(model_cls)
        except (TypeError, AttributeError):
            hints = getattr(model_cls, '__annotations__', {})
        
        for field_name, field_type in hints.items():
            full_field_name = f"{prefix}{field_name}" if prefix else field_name
            
            # Check if the field is an NDArray
            type_str = str(field_type)
            if ('NDArray' in type_str or 'ndarray' in type_str or 
                'numpy.typing' in type_str or 'np.float' in type_str or
                'npt.NDArray' in type_str):
                array_fields.append(full_field_name)
            
            # Check for List[float] which might represent arrays
            if (get_origin(field_type) == list or 
                'List[float]' in type_str or 
                'list[float]' in type_str):
                array_fields.append(full_field_name)
            
            # Check for nested models
            if get_origin(field_type) == list:
                args = get_args(field_type)
                if args and hasattr(args[0], '__annotations__'):
                    nested_model = args[0]
                    nested_fields = self._find_ndarray_fields(nested_model, f"{full_field_name}.")
                    array_fields.extend(nested_fields)
            elif hasattr(field_type, '__annotations__'):
                nested_fields = self._find_ndarray_fields(field_type, f"{full_field_name}.")
                array_fields.extend(nested_fields)
        
        return array_fields
    
    def _process_nested_value(self, value, array_fields: List[str], path: str = ''):
        """Process nested structures, converting lists to numpy arrays where appropriate"""
        if isinstance(value, dict):
            for k, v in value.items():
                current_path = f"{path}.{k}" if path else k
                if current_path in array_fields and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    value[k] = np.array(v, dtype=np.float32)
                else:
                    value[k] = self._process_nested_value(v, array_fields, current_path)
            return value
        elif isinstance(value, list):
            if path in array_fields and all(isinstance(x, (int, float)) for x in value):
                return np.array(value, dtype=np.float32)
            return [self._process_nested_value(item, array_fields, path) for item in value]
        else:
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
        
        # Find fields that should be numpy arrays
        array_fields = self._find_ndarray_fields(model_cls)
        
        # Get field types from the model
        field_types = {}
        try:
            model_annotations = get_type_hints(model_cls)
            for field_name, field_type in model_annotations.items():
                field_types[field_name] = field_type
        except Exception:
            pass
        
        try:
            result = []
            with open(cache_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    parsed_row = {}
                    for key, value in row.items():
                        if value == '':
                            parsed_row[key] = None
                            continue
                        
                        # Handle complex types stored as JSON
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                parsed_value = json.loads(value)
                                processed_value = self._process_nested_value(parsed_value, array_fields, key)
                                
                                # Special handling for response_segment with embeddings
                                if key == 'response_segment' and isinstance(processed_value, list):
                                    for segment in processed_value:
                                        if isinstance(segment, dict):
                                            for embed_field in ['code_embedding', 'description_embedding']:
                                                if embed_field in segment and isinstance(segment[embed_field], list):
                                                    segment[embed_field] = np.array(segment[embed_field], dtype=np.float32)
                                
                                parsed_row[key] = processed_value
                                continue
                            except (json.JSONDecodeError, TypeError):
                                pass
                        
                        # Handle primitive types
                        if value.lower() == 'true':
                            parsed_row[key] = True
                        elif value.lower() == 'false':
                            parsed_row[key] = False
                        else:
                            # Check if field should be string
                            should_be_string = False
                            if key in field_types:
                                field_type_str = str(field_types[key])
                                if ('str' in field_type_str or 'typing.Optional[str]' in field_type_str
                                    or key == 'response'):
                                    should_be_string = True
                            
                            if should_be_string:
                                parsed_row[key] = value
                            else:
                                try:
                                    if '.' in value:
                                        parsed_row[key] = float(value)
                                    else:
                                        parsed_row[key] = int(value)
                                except ValueError:
                                    parsed_row[key] = value
                    
                    # Create model instance
                    result.append(model_cls(**parsed_row))
            
            logger.info(f"Loaded {len(result)} items from cache for {filename} at step {step}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading cache for {filename} at step {step}: {e}")
            self.db.invalidate_cache(filename, step)
            return None
    
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
        
        # Check if we should use atomic writes
        import platform
        use_atomic = self.config.use_atomic_writes and platform.system() != 'Windows'
        
        if not use_atomic:
            # Direct write for Windows
            try:
                with open(cache_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Get field names from first item
                    sample = data[0]
                    serialize_method = getattr(sample, 'model_dump' if hasattr(sample, 'model_dump') else 'dict')
                    fieldnames = list(serialize_method().keys())
                    writer.writerow(fieldnames)
                    
                    # Custom JSON encoder for numpy types
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                                              np.int32, np.int64, np.uint8, np.uint16, 
                                              np.uint32, np.uint64)):
                                return int(obj)
                            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                                return float(obj)
                            if isinstance(obj, (np.bool_)):
                                return bool(obj)
                            return json.JSONEncoder.default(self, obj)
                    
                    # Write data rows
                    for item in data:
                        row = []
                        for field in fieldnames:
                            value = getattr(item, field)
                            
                            if value is None:
                                row.append('')
                                continue
                            
                            # Serialize complex types
                            def serialize_pydantic(obj):
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif hasattr(obj, 'model_dump'):
                                    return obj.model_dump()
                                elif hasattr(obj, 'dict'):
                                    return obj.dict()
                                elif isinstance(obj, list):
                                    return [serialize_pydantic(item) for item in obj]
                                elif isinstance(obj, dict):
                                    return {k: serialize_pydantic(v) for k, v in obj.items()}
                                else:
                                    return obj
                            
                            if isinstance(value, (list, dict)) or hasattr(value, 'dict') or hasattr(value, 'model_dump'):
                                serialized_value = serialize_pydantic(value)
                                row.append(json.dumps(serialized_value, cls=NumpyEncoder))
                            else:
                                # Handle numpy scalar types
                                if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                                                   np.int32, np.int64, np.uint8, np.uint16, 
                                                   np.uint32, np.uint64)):
                                    row.append(int(value))
                                elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                                    row.append(float(value))
                                elif isinstance(value, (np.bool_)):
                                    row.append(bool(value))
                                else:
                                    row.append(value)
                        
                        writer.writerow(row)
                
                # Calculate file hash
                file_hash = self._calculate_file_hash(cache_path)
                file_size = cache_path.stat().st_size
                
                # Record in database
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
                return False
        
        # Use atomic write with temporary file (non-Windows)
        temp_file = None
        try:
            # Create temporary file in same directory for atomic move
            temp_fd, temp_path = tempfile.mkstemp(
                dir=cache_path.parent, 
                prefix=f".{cache_path.stem}_", 
                suffix='.tmp'
            )
            temp_file = Path(temp_path)
            
            # Write to temporary file
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Get field names from first item
                sample = data[0]
                serialize_method = getattr(sample, 'model_dump' if hasattr(sample, 'model_dump') else 'dict')
                fieldnames = list(serialize_method().keys())
                writer.writerow(fieldnames)
                
                # Custom JSON encoder for numpy types
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                                          np.int32, np.int64, np.uint8, np.uint16, 
                                          np.uint32, np.uint64)):
                            return int(obj)
                        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                            return float(obj)
                        if isinstance(obj, (np.bool_)):
                            return bool(obj)
                        return json.JSONEncoder.default(self, obj)
                
                # Write data rows
                for item in data:
                    row = []
                    for field in fieldnames:
                        value = getattr(item, field)
                        
                        if value is None:
                            row.append('')
                            continue
                        
                        # Serialize complex types
                        def serialize_pydantic(obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif hasattr(obj, 'model_dump'):
                                return obj.model_dump()
                            elif hasattr(obj, 'dict'):
                                return obj.dict()
                            elif isinstance(obj, list):
                                return [serialize_pydantic(item) for item in obj]
                            elif isinstance(obj, dict):
                                return {k: serialize_pydantic(v) for k, v in obj.items()}
                            else:
                                return obj
                        
                        if isinstance(value, (list, dict)) or hasattr(value, 'dict') or hasattr(value, 'model_dump'):
                            serialized_value = serialize_pydantic(value)
                            row.append(json.dumps(serialized_value, cls=NumpyEncoder))
                        else:
                            # Handle numpy scalar types
                            if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                                               np.int32, np.int64, np.uint8, np.uint16, 
                                               np.uint32, np.uint64)):
                                row.append(int(value))
                            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                                row.append(float(value))
                            elif isinstance(value, (np.bool_)):
                                row.append(bool(value))
                            else:
                                row.append(value)
                    
                    writer.writerow(row)
            
            # Explicitly close the file handle
            os.close(temp_fd)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(temp_file)
            file_size = temp_file.stat().st_size
            
            # Move to final location - use platform-appropriate method
            if self.config.use_atomic_writes:
                try:
                    # Try atomic replace first
                    temp_file.replace(cache_path)
                except PermissionError:
                    # On Windows, close the file and try shutil.move
                    import platform
                    if platform.system() == 'Windows':
                        # Windows sometimes needs a moment to release file handles
                        import time
                        time.sleep(0.1)
                        shutil.move(str(temp_file), str(cache_path))
                    else:
                        raise
            else:
                shutil.move(str(temp_file), str(cache_path))
            
            # Record in database
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
            if temp_file and temp_file.exists():
                temp_file.unlink()
            return False
    
    def invalidate_cache(self, filename: str = None, step: str = None):
        """Invalidate cache entries"""
        self.db.invalidate_cache(filename, step)
    
    def cleanup_old_cache(self) -> int:
        """Remove old cache entries and files"""
        if not self.config.auto_cleanup:
            return 0
            
        # Get list of files to delete from database
        files_to_delete = self.db.cleanup_old_entries()
        
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
    
    def get_statistics(self) -> Dict:
        """Get cache usage statistics"""
        return self.db.get_cache_statistics()
    
    def cache_intermediate_data(self, data, filename: str, cache_key: str):
        """Cache intermediate processing data for phase-to-phase communication"""
        import pickle
        
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
        import pickle
        
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
    
