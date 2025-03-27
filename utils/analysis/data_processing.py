import numpy as np
import pandas as pd

def assign_confidence_bins(df, score_column="accuracy", n_bins=5):
    # Sort by the given score column
    df = df.sort_values(by=score_column).reset_index(drop=True)
    # Divide into n_bins equal-sized groups
    bin_size = len(df) // n_bins
    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i+1)*bin_size if i < n_bins - 1 else len(df)
        bin_df = df.iloc[start:end].copy()
        bin_df["confidence_level"] = i + 1  # bin 1, 2, 3, ...
        bins.append(bin_df)
    return pd.concat(bins, ignore_index=True)
