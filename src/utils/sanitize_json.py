import json
import os
import numpy as np
from typing import Any, Dict, List, Union


def sanitize_value(value: Any) -> Any:
    if value is None:
        return "NaN"
    
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return "NaN"
        return value
    
    if isinstance(value, (int, str, bool)):
        return value
    
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    
    # For any other type, try to convert to string or return as-is
    try:
        if hasattr(value, '__float__'):
            float_val = float(value)
            if np.isnan(float_val) or np.isinf(float_val):
                return "NaN"
    except (ValueError, TypeError):
        pass
    
    return value


def sanitize_json_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        return
    
    try:
        # Load JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Sanitize
        sanitized_data = sanitize_value(data)
        
        # Save back
        with open(file_path, 'w') as f:
            json.dump(sanitized_data, f, indent=2)
        
        print(f"Sanitized NaN values in {file_path}")
    except Exception as e:
        print(f"Warning: Could not sanitize {file_path}: {e}")


def sanitize_dict(data: Dict) -> Dict:
    return sanitize_value(data)

