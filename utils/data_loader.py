import json
import pandas as pd
import os

def load_dataset_results(dataset_name, model_name, sample_size, mode, base_path="./exp_results"):
    file_path = os.path.join(base_path, dataset_name, f"test_{sample_size}_perturbed_{mode}_{model_name}.jsonl")
    df = pd.read_json(file_path, lines=True)
    return df

def load_api_key(config_path="../api_key/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["openai_api_key"]

def load_precomputed_results(pickle_path):
    # Load a pickle file with precomputed bootstrap results
    import pickle
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data
