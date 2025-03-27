import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_bar_data(line_plot_data, model_names, columns, columns2names, source):
    
    # Update rcParams before plotting
    mpl.rcParams.update({
        'font.size': 18,        # base font size
        'axes.labelsize': 18,   # x and y labels
        'axes.titlesize': 18,   # subplot title
        'xtick.labelsize': 18,  # x tick labels
        'ytick.labelsize': 18,  # y tick labels
        'legend.fontsize': 20,  # legend
    })
    
    model_name_to_title = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "Llama-3.2-90B",
        "Qwen2-72B-Instruct": "Qwen2-72B",
        "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
        "gpt-4o-mini": "GPT-4o-mini",
        "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
        "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
        "Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B",
    }

    # Distinct color for each model
    model_colors = {
        "Llama-3.2-1B-Instruct":             "#1b9e77",
        "Llama-3.2-3B-Instruct":             "#d95f02",
        "Llama-3.2-11B-Vision-Instruct":     "#7570b3",
        "Llama-3.2-90B-Vision-Instruct-Turbo":   "#e7298a",
        "Qwen2-72B-Instruct":                "#66a61e",
        "Mixtral-8x7B-Instruct-v0.1":        "#e6ab02",
        "Mixtral-8x22B-Instruct-v0.1":       "#a6761d",
        "gpt-4o-mini":                       "#666666",
    }

    n_models = len(model_names)
    n_ability_levels = 5
    bar_width = 0.9

    # 2x4 grid for the 8 models
    fig, axes = plt.subplots(2, 4, figsize=(12, 12*0.618*0.7))
    axes = axes.ravel()  # Flatten for easy indexing
    pearson_pairs = {}

    for m, model_name in enumerate(model_names):
        ax = axes[m]

        if model_name not in line_plot_data or source not in line_plot_data[model_name]:
            raise ValueError(f"No line plot data for model {model_name_to_title.get(model_name, model_name)} and source {source}")

        wo_data = line_plot_data[model_name][source][columns[0]]  # Without misinformation
        w_data  = line_plot_data[model_name][source][columns[1]]  # With misinformation

        base_color = model_colors.get(model_name, "gray")

        for a in range(n_ability_levels):
            x_base = a

            wo_mean = wo_data[a]["mean"]
            wo_lb   = wo_data[a]["lb"]
            wo_ub   = wo_data[a]["ub"]

            w_mean  = w_data[a]["mean"]
            w_lb    = w_data[a]["lb"]
            w_ub    = w_data[a]["ub"]

            # Use alpha to indicate ability: darker for higher ability
            alpha = 1 - a * 0.18

            # Plot "with misinformation" (filled)
            ax.bar(x_base, w_mean, width=bar_width,
                   facecolor=base_color, alpha=alpha,
                   edgecolor='black',
                   yerr=[[w_mean - w_lb],[w_ub - w_mean]],
                   capsize=3, ecolor='black')

            # Overlay "without misinformation" (outline, no fill)
            ax.bar(x_base, wo_mean, width=bar_width,
                   facecolor='none', edgecolor=base_color,
                   linestyle='--', linewidth=2, alpha=alpha,
                   yerr=[[wo_mean - wo_lb],[wo_ub - wo_mean]],
                   capsize=3, ecolor=base_color)
            
            if model_name not in pearson_pairs:
                pearson_pairs[model_name] = []
            if wo_mean != 0:
                pearson_pairs[model_name].append({"x": x_base, "y": (wo_mean-w_mean)/wo_mean})

        ax.set_title(model_name_to_title.get(model_name, model_name), fontweight='bold')
        ax.set_xlim(-0.5, n_ability_levels - 0.5)
        # Use LaTeX \textsc{Accuracy} for the y-axis label
        ax.set_ylabel("K-Accuracy (%)", fontweight='bold')
        ax.set_xticks([])

    # Create the gradient strip for the legend (unchanged)
    gradient = np.linspace(0.0, 1.0, 21)  # from white (1) to black (0)
    gradient = np.vstack((gradient, gradient))
    gradient = np.tile(gradient, (4, 1))

    class HandlerImage(HandlerBase):
        def __init__(self, im_data, **kw):
            super().__init__(**kw)
            self.im_data = im_data

        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            offset_img = OffsetImage(self.im_data, cmap='gray', origin='lower', zoom=1.9)
            center_x = xdescent + width/2
            center_y = ydescent + height/2
            ab = AnnotationBbox(
                offset_img, (center_x, center_y),
                xycoords=trans, frameon=False, pad=0
            )
            return [ab]

    # Legend handles
    h_wo_legend = patches.Rectangle((0,0), 1, 1,
                                    fill=False, edgecolor='black',
                                    linestyle='--', linewidth=2)
    h_w_legend  = patches.Rectangle((0,0), 1, 1,
                                    facecolor='black', edgecolor='black')
    gradient_handle = object()

    # Single figure-level legend at the bottom
    legend = fig.legend(
        handles=[h_wo_legend, h_w_legend, gradient_handle],
        labels=["Original", "Misinformed", "Difficulty (lighter=less)"],
        handler_map={gradient_handle: HandlerImage(gradient)},
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),  # adjust as needed
        ncol=3
    )

    # Adjust layout so there's room at the bottom for the legend
    plt.tight_layout(rect=[0, 0.05, 1.05, 1.1])
    
    return pearson_pairs

def plot_bar_data_correction(df_results, model_names, columns, columns2names, source):
    # Update rcParams for uniform styling
    mpl.rcParams.update({
        'font.size': 16,        # Base font size
        'axes.labelsize': 14,   # X and Y labels
        'axes.titlesize': 14,   # Subplot title
        'xtick.labelsize': 14,  # X tick labels
        'ytick.labelsize': 14,  # Y tick labels
        'legend.fontsize': 14,  # Legend font size
    })
    
    model_name_to_title = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "Llama-3.2-90B",
        "Qwen2-72B-Instruct": "Qwen2-72B",
        "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
        "gpt-4o-mini": "GPT-4o-mini",
        "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
        "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
        "Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B",
    }

    # Distinct color for each model
    col_colors = {
        "prfx_q":                   "#1f77b4",
        "prfx_pert_prfx_q":         "#ff7f0e",
        "prfx_pert_prfx_q_both":    "#2ca02c",
        # "prfx_pert_prfx_q_2step":   "#d62728",
    }

    n_models = len(model_names)
    bar_width = 0.8

    # Create a 2x4 grid for the 8 models
    fig, axes = plt.subplots(2, 4, figsize=(9, 12*0.618*0.6), sharey=True)  # Adjust figure size
    axes = axes.ravel()  # Flatten for easy indexing

    for m, model_name in enumerate(model_names):
        ax = axes[m]

        if model_name not in df_results or source not in df_results[model_name]:
            raise ValueError(f"No data for model {model_name_to_title.get(model_name, model_name)} and source {source}")
        
        for c, column in enumerate(columns):
            base_color = col_colors.get(column, "gray")
            # Face color with alpha 0.6, edge color with alpha 1
            face_color = mpl.colors.to_rgba(base_color, alpha=0.6)
            edge_color = mpl.colors.to_rgba(base_color, alpha=1)
            mean = df_results[model_name][source]["mean"].loc["mix", column]
            margin = df_results[model_name][source]["margin"].loc["mix", column]
            lb = margin[0]
            ub = margin[1]

            # Plot "with misinformation" (filled)
            ax.bar(c, mean, width=bar_width,
                   facecolor=face_color,
                   edgecolor=edge_color, linewidth=1,
                   yerr=[[mean - lb], [ub - mean]],
                   capsize=3, ecolor=base_color,
                   label=columns2names.get(column, column))
            
            # Add mean values above the bars
            # ax.text(c, mean, f"{mean:.2f}", ha='center', va='bottom', fontsize=10)

        ax.set_xlabel(model_name_to_title.get(model_name, model_name), fontweight='bold')
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylabel("K-Accuracy (%)")
        ax.set_xticks([])

    # Add a single legend at the bottom center of the figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(columns), bbox_to_anchor=(0.5, 1.07))

    # Adjust layout to make space for the legend
    fig.tight_layout(rect=[0, 0, 1, 1])  # Leave space at the bottom for the legend