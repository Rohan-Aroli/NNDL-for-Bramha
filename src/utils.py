import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, float) and not np.isfinite(obj):
            return None # Handle infinity and NaN
        return super(NumpyEncoder, self).default(obj)

def save_json(data, filepath):
    """Saves a dictionary to a JSON file, handling numpy types correctly."""
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)
