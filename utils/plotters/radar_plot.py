import matplotlib.pyplot as plt
import numpy as np
from math import pi
import matplotlib as mpl

def create_radar_plots(
    df_results,               # a dict keyed by model_name -> {"mean": DataFrame, "margin": DataFrame}
    model_names,              # list of 8 models
    source,                   # "self" or "human"
    columns,                  # list of columns to compare (e.g. prfx_q, prfx_pert_prfx_q, etc.)
    columns2names,           # mapping column -> human-friendly label
    model_name_to_title,      # mapping model key -> short name (for x-label/spokes)
):
    
    # Update rcParams before plotting
    mpl.rcParams.update({
        'font.size': 18,        # base font size
        'axes.labelsize': 18,   # x and y labels
        'axes.titlesize': 18,   # subplot title
        'xtick.labelsize': 18,  # x tick labels
        'ytick.labelsize': 18,  # y tick labels
        'legend.fontsize': 12,  # legend
    })
    
    # We have 8 "spokes" – one per model
    N = len(model_names)

    # Compute the angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # repeat the first angle at the end to close the circle

    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(8, 8))

    # Make the radar start at the top and go clockwise
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set the spoke labels (the 8 models)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([model_name_to_title.get(m, m) for m in model_names])

    # Set the radial limits and ticks
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # Plot one line per column
    for col in columns:
        # Gather the mean & margin for each model, only for dataset="mix"
        vals = []
        lb   = []
        ub   = []

        for m in model_names:
            # Grab the mean for the single dataset "mix"
            mean_val = df_results[m][source]["mean"].loc["mix", col]
            margin_val = df_results[m][source]["margin"].loc["mix", col]  # e.g. [lower_bound, upper_bound]
            vals.append(mean_val)
            lb.append(margin_val[0])
            ub.append(margin_val[1])

        # Repeat the first value at the end to close the radial polygon
        vals += vals[:1]
        lb   += lb[:1]
        ub   += ub[:1]

        # Plot the mean line
        line = ax.plot(angles, vals, linewidth=1, linestyle='solid', 
                       label=columns2names.get(col, col))

        # Fill between lb and ub
        ax.fill_between(angles, lb, ub, alpha=0.2, color=line[0].get_color())
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15), handlelength=1.0, columnspacing=0.8)  # Adjust as desired