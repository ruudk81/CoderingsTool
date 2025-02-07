"""Cache manager to replace CSV handler with SQLite-backed caching"""

import csv
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Type, TypeVar, List, Optional, Dict, get_type_hints, get_origin, get_args
from pydantic import BaseModel
import tempfile
import shutil
import numpy as np

from cache_config import CacheConfig, ProcessingConfig
from cache_database import CacheDatabase


logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


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
    
    def cleanup_old_cache(self):
        """Remove old cache entries and files"""
        if self.config.auto_cleanup:
            deleted_count = self.db.cleanup_old_entries()
            logger.info(f"Cleaned up {deleted_count} old cache files")
    
    def get_statistics(self) -> Dict:
        """Get cache usage statistics"""
        return self.db.get_cache_statistics()
    
