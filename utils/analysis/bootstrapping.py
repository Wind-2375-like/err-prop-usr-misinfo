import numpy as np

def bootstrap(df, f, n=1000):
    seed = 42
    seeds = [seed + i for i in range(n)]
    total = len(df)
    ratios = []
    for i in range(n):
        sample = df.sample(total, replace=True, random_state=seeds[i])
        ratios.append(f(sample)*100)
    mean_ratio = np.mean(ratios)
    lower_bound = np.percentile(ratios, 2.5)
    upper_bound = np.percentile(ratios, 97.5)
    return mean_ratio, lower_bound, upper_bound

def bootstrap_with_ratios(df, f, n=1000):
    seed = 42
    seeds = [seed + i for i in range(n)]
    total = len(df)
    ratios = []
    for i in range(n):
        sample = df.sample(total, replace=True, random_state=seeds[i])
        ratios.append(f(sample)*100)
    return ratios
