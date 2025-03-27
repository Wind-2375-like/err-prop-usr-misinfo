import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_violin_data(violin_plot_data, model_name, columns2names, source, ax):
    # violin_plot_data[model_name][source][col_key] = { "ratios": [...], "mean": val, "lb": val, "ub": val }
    # We assume accuracy=1.0 subset used for these computations
    # We'll create a violin plot of ratios for each column in columns2names.
    model_name_to_title = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "LlaMAV-3.2-90B",
        "Qwen2-72B-Instruct": "Qwen2-72B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
        "gpt-4o-mini": "GPT-4o-mini",
    }

    if model_name not in violin_plot_data:
        raise ValueError(f"No violin data for model {model_name}")

    if source not in violin_plot_data[model_name]:
        raise ValueError(f"No source {source} data for model {model_name}")

    data = []
    labels = []
    for col_key, col_name in columns2names.items():
        vals = violin_plot_data[model_name][source][col_key]["ratios"]
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(col_name)

    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return

    sns.violinplot(data=data, ax=ax, inner='quartile')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(20, 100)
    ax.set_ylabel("Subset Accuracy Under Misinformation (%)")
    ax.set_title(f"{model_name_to_title.get(model_name, model_name)}")
