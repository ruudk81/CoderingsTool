import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import os
import csv
import json
from typing import Type, TypeVar, List, get_type_hints, get_origin, get_args
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class CsvHandler:
    def __init__(self, data_dir: str = None):
        current_dir = os.getcwd()
        if os.path.basename(current_dir) == 'utils':
            data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'data'))
        elif os.path.basename(current_dir) == 'modules':
            data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
        else:
            data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))  
        self.data_dir = data_dir
    
    def get_filepath(self, filename: str, extension: str):
        if filename.endswith('.csv'):
            return os.path.join(self.data_dir, filename)
        new_filename = filename.replace('.sav', f'_{extension}.csv')
        return os.path.join(self.data_dir, new_filename)
    
    def save_to_csv(self, data_list: List[T], filename: str, extension: str):
        if not data_list:
            return
        
        # Import numpy here to avoid making it a global dependency
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False
        
        filepath = self.get_filepath(filename, extension)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
        
            sample = data_list[0]
            # Get the appropriate serialization method based on Pydantic version
            serialize_method = getattr(sample, 'model_dump' if hasattr(sample, 'model_dump') else 'dict')
            fieldnames = list(serialize_method().keys())
            writer.writerow(fieldnames)
        
            for item in data_list:
                row = []
                for field in fieldnames:
                    value = getattr(item, field)
                    
                    if value is None:
                        row.append('')
                        continue
                    
                    def serialize_pydantic(obj):
                        # Add handling for numpy arrays
                        if HAS_NUMPY and isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif hasattr(obj, 'model_dump'):  # Pydantic v2
                            return obj.model_dump()
                        elif hasattr(obj, 'dict'):      # Pydantic v1
                            return obj.dict()
                        elif isinstance(obj, list):
                            return [serialize_pydantic(item) for item in obj]
                        elif isinstance(obj, dict):
                            return {k: serialize_pydantic(v) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Custom JSON encoder to handle numpy types
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if HAS_NUMPY:
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
                    
                    if isinstance(value, (list, dict)) or hasattr(value, 'dict') or hasattr(value, 'model_dump'):
                        try:
                            serialized_value = serialize_pydantic(value)
                            # Use the custom encoder for JSON serialization
                            row.append(json.dumps(serialized_value, cls=NumpyEncoder))
                        except TypeError as e:
                            # Fallback in case of serialization errors
                            if HAS_NUMPY and 'numpy' in str(e).lower():
                                # Try again with explicit conversion
                                serialized_value = serialize_pydantic(value)
                                if isinstance(serialized_value, dict):
                                    for k, v in serialized_value.items():
                                        if HAS_NUMPY and isinstance(v, np.ndarray):
                                            serialized_value[k] = v.tolist()
                                row.append(json.dumps(serialized_value, cls=NumpyEncoder))
                            else:
                                # Re-raise if it's not a numpy-related error
                                raise
                    else:
                        # Special handling for numpy scalar types
                        if HAS_NUMPY:
                            if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                                                np.int32, np.int64, np.uint8, np.uint16, 
                                                np.uint32, np.uint64)):
                                row.append(int(value))
                                continue
                            if isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                                row.append(float(value))
                                continue
                            if isinstance(value, (np.bool_)):
                                row.append(bool(value))
                                continue
                        
                        row.append(value)
                
                writer.writerow(row)
    
    def _find_ndarray_fields(self, model_cls: Type[BaseModel], prefix: str = '') -> List[str]:
        """
        Recursively find all fields that are numpy ndarrays or contain them in nested models.
        """
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False
            return []  # If numpy isn't available, we can't identify ndarray fields
        
        array_fields = []
        
        # Get type hints for the model
        try:
            hints = get_type_hints(model_cls)
        except (TypeError, AttributeError):
            hints = getattr(model_cls, '__annotations__', {})
        
        for field_name, field_type in hints.items():
            full_field_name = f"{prefix}{field_name}" if prefix else field_name
            
            # Check if the field is an NDArray directly
            is_ndarray = False
            type_str = str(field_type)
            if ('NDArray' in type_str or 'ndarray' in type_str or 
                'numpy.typing' in type_str or 'np.float' in type_str or
                'npt.NDArray' in type_str):
                array_fields.append(full_field_name)
                is_ndarray = True
            
            # Handle typing.List[float] or list[float] which might represent arrays
            if not is_ndarray and (get_origin(field_type) == list or 
                                  'List[float]' in type_str or 
                                  'list[float]' in type_str):
                array_fields.append(full_field_name)
            
            # Check for List of nested models that might contain NDArrays
            if get_origin(field_type) == list:
                args = get_args(field_type)
                if args and hasattr(args[0], '__annotations__'):
                    # This is a List of some model class
                    nested_model = args[0]
                    nested_fields = self._find_ndarray_fields(nested_model, f"{full_field_name}.")
                    array_fields.extend(nested_fields)
            
            # Check if it's a nested model directly
            elif hasattr(field_type, '__annotations__'):
                nested_fields = self._find_ndarray_fields(field_type, f"{full_field_name}.")
                array_fields.extend(nested_fields)
        
        return array_fields
    
    def _process_nested_value(self, value, array_fields: List[str], path: str = ''):
        """
        Process a nested structure, converting lists to numpy arrays where appropriate.
        """
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False
            return value
            
        if not HAS_NUMPY:
            return value
            
        if isinstance(value, dict):
            for k, v in value.items():
                current_path = f"{path}.{k}" if path else k
                # Check if this is an array field
                if current_path in array_fields and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    value[k] = np.array(v, dtype=np.float32)
                else:
                    value[k] = self._process_nested_value(v, array_fields, current_path)
            return value
        elif isinstance(value, list):
            # Check if this path directly matches an array field
            if path in array_fields and all(isinstance(x, (int, float)) for x in value):
                return np.array(value, dtype=np.float32)
                
            # Process each item in the list
            return [self._process_nested_value(item, array_fields, path) for item in value]
        else:
            return value
    
    def load_from_csv(self, filename: str, extension: str, model_cls: Type[T], nested_fields: dict = None) -> List[T]:    
        filepath = self.get_filepath(filename, extension)
        csv.field_size_limit(2**31 - 1)   
        
        if nested_fields is None:
            nested_fields = {}
    
        # Import numpy here for array conversion
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False
        
        # Find all fields that should be ndarrays
        array_fields = self._find_ndarray_fields(model_cls)
        
        # Get field types from the model to respect during conversion
        field_types = {}
        try:
            model_annotations = get_type_hints(model_cls)
            for field_name, field_type in model_annotations.items():
                field_types[field_name] = field_type
        except Exception:
            # If we can't get type hints, we'll use a more basic approach
            pass
        
        result = []
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                parsed_row = {}
                for key, value in row.items():
                    if value == '':
                        parsed_row[key] = None
                        continue
                    
                    # Handle complex types that are stored as JSON
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            parsed_value = json.loads(value)
                            
                            # Process nested structures to convert arrays
                            processed_value = self._process_nested_value(parsed_value, array_fields, key)
                            
                            # Handle nested models with NDArrays specifically for response_segment
                            if key == 'response_segment' and isinstance(processed_value, list):
                                # Import numpy for conversion
                                if HAS_NUMPY:
                                    # Process each segment in the list
                                    for segment in processed_value:
                                        # Convert embeddings fields to numpy arrays
                                        if isinstance(segment, dict):
                                            for embed_field in ['code_embedding', 'description_embedding']:
                                                if embed_field in segment and isinstance(segment[embed_field], list):
                                                    segment[embed_field] = np.array(segment[embed_field], dtype=np.float32)
                            
                            # Handle nested Pydantic models
                            if key in nested_fields and isinstance(processed_value, list):
                                submodel_cls = nested_fields[key]
                                if submodel_cls:
                                    processed_value = [submodel_cls(**item) for item in processed_value]
                            
                            parsed_row[key] = processed_value
                            continue
                        except (json.JSONDecodeError, TypeError) as e:
                            # Not valid JSON, treat as regular value
                            pass
                    
                    # Handle primitive types
                    if value.lower() == 'true':
                        parsed_row[key] = True
                    elif value.lower() == 'false':
                        parsed_row[key] = False
                    else:
                        # Check if this field should be a string based on the model definition
                        should_be_string = False
                        
                        if key in field_types:
                            # Check if field type is str or Optional[str]
                            field_type_str = str(field_types[key])
                            if ('str' in field_type_str or 'typing.Optional[str]' in field_type_str
                                or key == 'response'):  # Special case for 'response' field
                                should_be_string = True
                        
                        if should_be_string:
                            # Keep as string if the model expects a string
                            parsed_row[key] = value
                        else:
                            try:
                                # Try to convert to number if it looks like one
                                if '.' in value:
                                    parsed_row[key] = float(value)
                                else:
                                    parsed_row[key] = int(value)
                            except ValueError:
                                # If conversion fails, keep as string
                                parsed_row[key] = value
                
                # Direct conversion for EmbeddingsModel before model creation
                if model_cls.__name__ == 'EmbeddingsModel' and HAS_NUMPY and 'response_segment' in parsed_row:
                    if isinstance(parsed_row['response_segment'], list):
                        for segment in parsed_row['response_segment']:
                            # Ensure these are numpy arrays before model validation
                            for field in ['code_embedding', 'description_embedding']:
                                if field in segment and isinstance(segment[field], list):
                                    segment[field] = np.array(segment[field], dtype=np.float32)
                
                # Create model instance with properly typed data
                try:
                    result.append(model_cls(**parsed_row))
                except Exception as e:
                    # Add more debugging info
                    print(f"Error creating model from row: {e}")
                    print(f"Problematic row: {parsed_row}")
                    if model_cls.__name__ == 'EmbeddingsModel' and 'response_segment' in parsed_row:
                        for i, segment in enumerate(parsed_row.get('response_segment', [])):
                            for field in ['code_embedding', 'description_embedding']:
                                if field in segment:
                                    print(f"Field {field} in segment {i} has type: {type(segment[field])}")
                    # You could log the problematic row here
                    raise
        
        return result
        
    # def load_from_csv(self, filename: str, extension: str, model_cls: Type[T], nested_fields: dict = None) -> List[T]:    
    #     filepath = self.get_filepath(filename, extension)
    #     csv.field_size_limit(2**31 - 1)   
        
    #     if nested_fields is None:
    #         nested_fields = {}
            
    #     # For EmbeddingsModel, automatically set the nested field for response_segment
    #     if model_cls.__name__ == 'EmbeddingsModel' and 'response_segment' not in nested_fields:
    #         # Attempt to find the EmbeddingsSubmodel in the same module
    #         module = getattr(model_cls, '__module__', None)
    #         if module:
    #             try:
    #                 import sys
    #                 if module in sys.modules:
    #                     module_obj = sys.modules[module]
    #                     if hasattr(module_obj, 'EmbeddingsSubmodel'):
    #                         nested_fields['response_segment'] = module_obj.EmbeddingsSubmodel
    #             except Exception:
    #                 pass  # If it fails, we'll proceed without automatic mapping
        
    #     # Import numpy here for array conversion
    #     try:
    #         import numpy as np
    #         HAS_NUMPY = True
    #     except ImportError:
    #         HAS_NUMPY = False
        
    #     # Find all fields that should be ndarrays
    #     array_fields = self._find_ndarray_fields(model_cls)
        
    #     result = []
    #     with open(filepath, 'r', newline='', encoding='utf-8') as f:
    #         reader = csv.DictReader(f)
            
    #         for row in reader:
    #             parsed_row = {}
    #             for key, value in row.items():
    #                 if value == '':
    #                     parsed_row[key] = None
    #                     continue
                    
    #                 # Handle complex types that are stored as JSON
    #                 if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
    #                     try:
    #                         parsed_value = json.loads(value)
                            
    #                         # Process nested structures to convert arrays
    #                         processed_value = self._process_nested_value(parsed_value, array_fields, key)
                            
    #                         # Handle nested models with NDArrays specifically for response_segment
    #                         if key == 'response_segment' and isinstance(processed_value, list):
    #                             # Import numpy for conversion
    #                             if HAS_NUMPY:
    #                                 import numpy as np
    #                                 # Process each segment in the list
    #                                 for segment in processed_value:
    #                                     # Convert embeddings fields to numpy arrays
    #                                     if isinstance(segment, dict):
    #                                         for embed_field in ['code_embedding', 'description_embedding']:
    #                                             if embed_field in segment and isinstance(segment[embed_field], list):
    #                                                 segment[embed_field] = np.array(segment[embed_field], dtype=np.float32)
                            
    #                         # Handle nested Pydantic models
    #                         if key in nested_fields and isinstance(processed_value, list):
    #                             submodel_cls = nested_fields[key]
    #                             if submodel_cls:
    #                                 processed_value = [submodel_cls(**item) for item in processed_value]
                            
    #                         parsed_row[key] = processed_value
    #                         continue
    #                     except (json.JSONDecodeError, TypeError) as e:
    #                         # Not valid JSON, treat as regular value
    #                         pass
                    
    #                 # Handle primitive types
    #                 if value.lower() == 'true':
    #                     parsed_row[key] = True
    #                 elif value.lower() == 'false':
    #                     parsed_row[key] = False
    #                 else:
    #                     try:
    #                         # Try to convert to number if it looks like one
    #                         if '.' in value:
    #                             parsed_row[key] = float(value)
    #                         else:
    #                             parsed_row[key] = int(value)
    #                     except ValueError:
    #                         # If conversion fails, keep as string
    #                         parsed_row[key] = value
                
    #             # Direct conversion for EmbeddingsModel before model creation
    #             if model_cls.__name__ == 'EmbeddingsModel' and HAS_NUMPY and 'response_segment' in parsed_row:
    #                 if isinstance(parsed_row['response_segment'], list):
    #                     for segment in parsed_row['response_segment']:
    #                         # Ensure these are numpy arrays before model validation
    #                         for field in ['code_embedding', 'description_embedding']:
    #                             if field in segment and isinstance(segment[field], list):
    #                                 segment[field] = np.array(segment[field], dtype=np.float32)
                
    #             # Create model instance with properly typed data
    #             try:
    #                 result.append(model_cls(**parsed_row))
    #             except Exception as e:
    #                 # Add more debugging info
    #                 print(f"Error creating model from row: {e}")
    #                 if model_cls.__name__ == 'EmbeddingsModel' and 'response_segment' in parsed_row:
    #                     for i, segment in enumerate(parsed_row.get('response_segment', [])):
    #                         for field in ['code_embedding', 'description_embedding']:
    #                             if field in segment:
    #                                 print(f"Field {field} in segment {i} has type: {type(segment[field])}")
    #                 # You could log the problematic row here
    #                 raise
        
    #     return result