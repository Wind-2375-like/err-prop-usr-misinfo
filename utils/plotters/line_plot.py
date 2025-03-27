import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_line_data(line_plot_data, model_name, columns, columns2names, source, ax):
    # Update rcParams before plotting
    mpl.rcParams.update({
        'font.size': 16,        # base font size
        'axes.labelsize': 12,   # x and y labels
        'axes.titlesize': 12,   # subplot title
        'xtick.labelsize': 12,  # x tick labels
        'ytick.labelsize': 12,  # y tick labels
        'legend.fontsize': 12,  # legend
    })
    
    # line_plot_data[model_name][column] = list of dicts:
    # each dict: {"confidence_level": cl, "mean": mean_val, "lb": lb_val, "ub": ub_val, "ratios": ratios}
    # We assume data already loaded from line_plot_data.pkl
    model_name_to_title = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "LlaMAV-3.2-90B",
        "Qwen2-72B-Instruct": "Qwen2-72B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
        "gpt-4o-mini": "GPT-4o-mini",
    }

    if model_name not in line_plot_data:
        raise ValueError(f"No line plot data for model {model_name_to_title.get(model_name, model_name)}")

    for column in columns:
        bin_data = line_plot_data[model_name][source][column]
        cl_levels = [d["confidence_level"] for d in bin_data]
        means = [d["mean"] for d in bin_data]
        lbs = [d["lb"] for d in bin_data]
        ubs = [d["ub"] for d in bin_data]

        # Filter out NaNs
        ax.plot(cl_levels, means, marker='o', label=columns2names.get(column, column))
        ax.fill_between(cl_levels, lbs, ubs, alpha=0.2)

    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Subset Accuracy (%)")
    ax.set_title(f"{model_name}")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True)
    ax.legend(loc='lower right')

def plot_line_data_for_model_size(line_plot_data, model_names, columns, columns2names, source):
     # Update rcParams before plotting
    mpl.rcParams.update({
        'font.size': 16,        # base font size
        'axes.labelsize': 24,   # x and y labels
        'axes.titlesize': 20,   # subplot title
        'xtick.labelsize': 24,  # x tick labels
        'ytick.labelsize': 24,  # y tick labels
        'legend.fontsize': 18,  # legend
    })
    
    # line_plot_data[model_name][column] = list of dicts:
    # each dict: {"confidence_level": cl, "mean": mean_val, "lb": lb_val, "ub": ub_val, "ratios": ratios}
    # We assume data already loaded from line_plot_data.pkl
    model_name_to_title = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "90B",
        "Llama-3.2-1B-Instruct": "1B",
        "Llama-3.2-3B-Instruct": "3B",
        "Llama-3.2-11B-Vision-Instruct": "11B",
    }

    xs = [1, 3, 11, 90]
    for column in columns:
        means = [line_plot_data[model_name][source]["mean"].loc['mix'][column] for model_name in model_names]
        margins = [line_plot_data[model_name][source]["margin"].loc['mix'][column] for model_name in model_names]
        lbs = [m[0] for m in margins]
        ubs = [m[1] for m in margins]

        # Filter out NaNs
        plt.plot(xs, means, marker='o', label=columns2names.get(column, column))
        plt.fill_between(xs, lbs, ubs, alpha=0.2)

    ax = plt.gca()
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Subset Accuracy (%)")
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 11, 90])
    ax.set_xticklabels([model_name_to_title.get(model_name, model_name) for model_name in model_names])
    ax.grid(True)
    ax.legend(loc='lower right', ncol=4)