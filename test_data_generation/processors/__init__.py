from test_data_generation.processors.dataset_gsm8k import DatasetGsm8kProcessor
from test_data_generation.processors.dataset_math import DatasetMathProcessor
from test_data_generation.processors.dataset_mathqa import DatasetMathqaProcessor
from test_data_generation.processors.dataset_metamath import DatasetMetamathProcessor

def get_processor(dataset_name):
    if dataset_name == "gsm8k":
        return DatasetGsm8kProcessor(dataset_name)
    elif dataset_name == "math":
        return DatasetMathProcessor(dataset_name)
    elif dataset_name == "mathqa":
        return DatasetMathqaProcessor(dataset_name)
    elif dataset_name == "metamath":
        return DatasetMetamathProcessor(dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
