import pandas as pd
import json

def read_jsonl_file(file_path):
    """Reads a JSONL file into a Pandas DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)
