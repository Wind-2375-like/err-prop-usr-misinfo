import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
import matplotlib as mpl

# Define colors for each category
COLOR_BASELINE = '#1f77b4'   # W.o. misinformation
COLOR_MISINFO  = '#ff7f0e'   # W. misinformation
COLOR_CORR_SC  = 'green'   
COLOR_CORR_FC  = 'red'   
COLOR_CORR_NC  = '#ba8e23'   

# Hatch pattern for "Error detection only"
HATCH_PQ = '///'

def plot_error_analysis_bar(error_analysis_data, ax, middle=False):
    """
    Plots a bar chart with baseline lines for a single error_analysis_data dictionary.
    """
    if not error_analysis_data:
        ax.text(0.5, 0.5, 'No Error Analysis Data Available',
                ha='center', va='center')
        return
    
    y_max = 0
    y_min = 100
    
    baseline_mean, baseline_lb, baseline_ub = error_analysis_data["baseline"]
    perturb_baseline_mean, perturb_baseline_lb, perturb_baseline_ub = error_analysis_data["perturb_baseline"]
    
    if baseline_ub > y_max:
        y_max = baseline_ub
    if baseline_lb < y_min:
        y_min = baseline_lb
    if perturb_baseline_ub > y_max:
        y_max = perturb_baseline_ub
    if perturb_baseline_lb < y_min:
        y_min = perturb_baseline_lb
    
    position_order = error_analysis_data["position_order"]
    position_mapping = {label: idx for idx, label in enumerate(position_order)}

    results = error_analysis_data["main_results"]
    additional_results = error_analysis_data["additional_results"]

    df_main = results
    df_add = additional_results
    bar_width = 0.9

    sns.set_style("whitegrid")

    # main_data = EDC across positions
    main_data = [r for r in df_main if r['label'] == 'Perturbation Positions']
    if len(main_data) == 0:
        ax.text(0.5, 0.5, 'No Main Results Available',
                ha='center', va='center')
        return
    
    # We specifically want main_data[1] -> "Error correction (Perfect Quality)"
    if len(main_data) < 2:
        ax.text(0.5, 0.5,
                'Not enough main_data entries (need at least 2).',
                ha='center', va='center')
        return

    # The bar for "Error correction (Perfect Quality)"
    edc_mean = main_data[1]['mean_ratio']
    edc_lb   = main_data[1]['ci'][0]
    edc_ub   = main_data[1]['ci'][1]
    if edc_ub > y_max:
        y_max = edc_ub
    if edc_lb < y_min:
        y_min = edc_lb
    
    # Plot the bar at x=0
    ax.bar(
        0, edc_mean, 
        width=bar_width,
        facecolor=COLOR_CORR_SC,
        edgecolor=COLOR_CORR_SC,
        ecolor=COLOR_CORR_SC,
        linewidth=1.5, alpha=0.6,
        yerr=[[edc_mean - edc_lb],[edc_ub - edc_mean]],
        capsize=3,
        label='F-Corr'
    )

    labels_seen = set()
    # Plot additional bars from df_add
    for i, d in enumerate(df_add):
        y = d['mean_ratio']
        lb = d['ci'][0]
        ub = d['ci'][1]
        label_original = d['label']
        if ub > y_max:
            y_max = ub
        if lb < y_min:
            y_min = lb

        # Decide color/hatch based on label
        if label_original == 'Error Detection and Correction (Perfect Quality)':
            continue  # Already plotted
        
        if label_original == 'Error Detection and Correction (Bad Quality)':
            color = COLOR_CORR_FC
            new_label = 'NF-Corr'
            facecolor = color
            edgecolor = color
            hatch     = None

        elif label_original == 'Error Detection Only (Perfect Quality)':
            continue

        elif label_original == 'Error Detection Only (Bad Quality)':
            color = COLOR_CORR_NC
            new_label = 'N-Corr'
            facecolor = color
            edgecolor = color
            hatch     = None

        else:
            # Not needed or skip
            continue
        
        ax.bar(
            i+1 if i == 0 else i,
            y,
            width=bar_width,
            facecolor=facecolor,
            edgecolor=edgecolor,
            ecolor=edgecolor,
            hatch=hatch,
            linewidth=1.5, alpha=0.6,
            yerr=[[y - lb],[ub - y]],
            capsize=3,
            label=(new_label if new_label not in labels_seen else None)
        )
        labels_seen.add(new_label)
        
    # Force x-limit to show up to ~3 on x-axis
    last_bar_index = 3
    ax.set_xlim(-0.5, last_bar_index + 0.5 - 1)

    # Draw horizontal lines for baseline
    xmin, xmax = ax.get_xlim()
    ax.axhline(
        y=baseline_mean, color=COLOR_BASELINE, linestyle='--', linewidth=2,
        label='Original' if 'W.o. misinformation' not in labels_seen else None
    )
    ax.fill_between(
        [xmin, xmax],
        [baseline_lb, baseline_lb], [baseline_ub, baseline_ub],
        color=COLOR_BASELINE, alpha=0.2
    )
    labels_seen.add('W.o. misinformation')

    # Baseline with misinformation
    ax.axhline(
        y=perturb_baseline_mean, color=COLOR_MISINFO, linestyle='--', linewidth=2,
        label='Misinformed' if 'W. misinformation' not in labels_seen else None
    )
    ax.fill_between(
        [xmin, xmax],
        [perturb_baseline_lb, perturb_baseline_lb],
        [perturb_baseline_ub, perturb_baseline_ub],
        color=COLOR_MISINFO, alpha=0.2
    )
    labels_seen.add('W. misinformation')

    ax.set_ylim(20, 90)

    ax.set_xlabel('Position at 0%', fontweight='bold')
    if middle:
        ax.set_ylabel('K-Accuracy (%)', fontweight='bold')
    ax.set_xticks([])
    ax.grid(True)

def plot_error_analysis_line(error_analysis_data, ax, middle=False):
    """
    Plots a line chart (only main data + baseline lines) for a single error_analysis_data dictionary.
    """
    if not error_analysis_data:
        ax.text(0.5, 0.5, 'No Error Analysis Data Available',
                ha='center', va='center')
        return
    
    y_max = 0
    y_min = 100

    baseline_mean, baseline_lb, baseline_ub = error_analysis_data["baseline"]
    perturb_baseline_mean, perturb_baseline_lb, perturb_baseline_ub = error_analysis_data["perturb_baseline"]
    
    if baseline_ub > y_max:
        y_max = baseline_ub
    if baseline_lb < y_min:
        y_min = baseline_lb
    if perturb_baseline_ub > y_max:
        y_max = perturb_baseline_ub
    if perturb_baseline_lb < y_min:
        y_min = perturb_baseline_lb

    position_order = error_analysis_data["position_order"][1:]
    position_mapping = {label: idx for idx, label in enumerate(position_order)}
    df_main = error_analysis_data["main_results"][1:]
    sns.set_style("whitegrid")

    labels_seen = set()

    # Plot main line => "Error correction (Perfect Quality)"
    main_data = [r for r in df_main if r['label'] == 'Perturbation Positions']
    if len(main_data) > 0:
        main_data_sorted = sorted(
            main_data, key=lambda x: position_mapping[x['position']]
        )
        x_main = [position_mapping[d['position']] for d in main_data_sorted]
        y_main = [d['mean_ratio'] for d in main_data_sorted]
        lb_main = [d['ci'][0] for d in main_data_sorted]
        ub_main = [d['ci'][1] for d in main_data_sorted]
        
        if max(ub_main) > y_max:
            y_max = max(ub_main)
        if min(lb_main) < y_min:
            y_min = min(lb_main)

        ax.plot(
            x_main, y_main, color=COLOR_CORR_SC, alpha=0.8, marker='o',
            label='F-Corr'
        )
        ax.fill_between(
            x_main, lb_main, ub_main,
            color=COLOR_CORR_SC, alpha=0.2
        )
        labels_seen.add('F-Corr')

    # Fill across entire subplot
    ax.set_xlim(-0.5, len(position_order) - 0.5)
    xmin, xmax = ax.get_xlim()

    # W.o. misinformation
    ax.axhline(
        y=baseline_mean, color=COLOR_BASELINE, linestyle='--', linewidth=2,
        label='Original' if 'W.o. misinformation' not in labels_seen else None
    )
    ax.fill_between(
        [xmin, xmax],
        [baseline_lb, baseline_lb], [baseline_ub, baseline_ub],
        color=COLOR_BASELINE, alpha=0.2
    )
    labels_seen.add('W.o. misinformation')

    # W. misinformation
    ax.axhline(
        y=perturb_baseline_mean, color=COLOR_MISINFO, linestyle='--', linewidth=2,
        label='Misinformed' if 'W. misinformation' not in labels_seen else None
    )
    ax.fill_between(
        [xmin, xmax],
        [perturb_baseline_lb, perturb_baseline_lb],
        [perturb_baseline_ub, perturb_baseline_ub],
        color=COLOR_MISINFO, alpha=0.2
    )
    labels_seen.add('W. misinformation')

    ax.set_ylim(20, 90)

    ax.set_xlabel('Position', fontweight='bold')
    if middle:
        ax.set_ylabel('K-Accuracy (%)', fontweight='bold')
    ax.set_xticks(range(len(position_order)))
    ax.set_xticklabels(
        position_order, rotation=20, ha='center'
    )
    ax.grid(True)

def plot_error_analysis_full(error_analysis_data_list):
    """
    Plots error analyses in a grid of subplots, 2 columns wide:
      - Left column: bar plot + horizontal lines for baseline
      - Right column: line plot with baseline + main data
    Each row shares the same Y-axis.

    We make it "flatter" by using a smaller total height, a narrower
    bar subplot, and a wider line subplot. We place a single,
    combined legend at the top (or bottom) so that it includes 
    bar + line entries.
    """
    titles = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-11B"]
    # Update rcParams before plotting
    mpl.rcParams.update({
        'font.size': 20,        # base font size
        'axes.labelsize': 20,   # x and y labels
        'axes.titlesize': 20,   # subplot title
        'xtick.labelsize': 18,  # x tick labels
        'ytick.labelsize': 20,  # y tick labels
        'legend.fontsize': 18,  # legend
    })
    
    len_data = len(error_analysis_data_list)
    
    # You can tweak these to make the figure "flatter"
    fig, axs = plt.subplots(
        nrows=len_data, ncols=2,
        figsize=(12, 6),          # Width=12in, Height=4in
        sharey='row',
        gridspec_kw={'width_ratios': [0.5, 1]}  # narrower left, wider right
    )
    # Make spacing compact
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # Plot
    if len_data == 1:
        plot_error_analysis_bar(error_analysis_data_list[0], axs[0], middle=True)
        plot_error_analysis_line(error_analysis_data_list[0], axs[1], middle=True)
    else:
        for i, error_analysis_data in enumerate(error_analysis_data_list):
            # If we want the middle row to have Y-label, do:
            middle = (i == len_data // 2)
            plot_error_analysis_bar(error_analysis_data, axs[i, 0], middle=middle)
            plot_error_analysis_line(error_analysis_data, axs[i, 1], middle=middle)
            axs[i, 0].set_title(f"{titles[i]}", fontweight='bold')
            axs[i, 1].set_title(f"{titles[i]}", fontweight='bold')
            
            # Remove x-axis labels for all but the last row (makes it more compact)
            if i < len_data - 1:
                axs[i, 0].set_xlabel('')
                axs[i, 0].set_xticklabels([])
                axs[i, 1].set_xlabel('')
                axs[i, 1].set_xticklabels([])

    # =============== COMBINED LEGEND ===============
    # Instead of removing legends in each subplot, we can gather them from *all* subplots:
    handles, labels = [], []
    for ax_row in axs: 
        # If there's only one row, ax_row might be Axes, not a list
        if isinstance(ax_row, np.ndarray):
            # We have subplots in a row
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                for (handle, label) in zip(h, l):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
        else:
            # Single row case: ax_row is a single Axes
            h, l = ax_row.get_legend_handles_labels()
            for (handle, label) in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)

    # Now place the combined legend at the top center or bottom center
    
    fig.legend(
        handles, labels,
        loc='upper center',      # or 'lower center'
        ncol=5,                  # 5 columns
        bbox_to_anchor=(0.5, 1.05),  # move it above the subplot area
        frameon=False,
    )

    plt.tight_layout()
    
def plot_error_analysis_full_line(error_analysis_data_list):
    """
    Creates a figure with 3 rows, 1 column of line plots.
    Each subplot uses plot_error_analysis_line to show the error analysis,
    and a combined legend is placed at the top.
    """
    # Use your preferred model names (same as before)
    titles = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-11B"]
    len_data = len(error_analysis_data_list)
    
    # Create figure: 3 rows, 1 column; share y-axis among subplots.
    fig, axs = plt.subplots(nrows=len_data, ncols=1, figsize=(8, 6), sharey=True)
    # Ensure axs is iterable (even if there is only one subplot)
    if len_data == 1:
        axs = [axs]
    
    # Plot each error analysis data as a line plot.
    for i, (data, ax) in enumerate(zip(error_analysis_data_list, axs)):
        # Pass the "middle" flag if desired (here, the middle row will have a y-label)
        middle = (i == len_data // 2)
        plot_error_analysis_line(data, ax, middle=middle)
        # Set the subplot title to the model name (optional if you want it at the top)
        ax.set_title(titles[i], fontweight='bold')
        # Remove x-axis labels for all but the last row (makes it more compact)
        if i < len_data - 1:
            axs[i].set_xlabel('')
            axs[i].set_xticklabels([])
    
    # Collect legend handles and labels from all subplots.
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    # Place the combined legend at the top center.
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False
    )
    
    plt.tight_layout()
    

def plot_error_analysis_full_bar(error_analysis_data_list):
    """
    Creates a figure with 1 row, 3 columns of bar plots.
    The subplots share the same y-axis; x–axis labels are removed,
    and each subplot displays its model name underneath.
    """
    mpl.rcParams.update({
        'legend.fontsize': 13,  # legend
    })
    
    titles = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-11B"]
    len_data = len(error_analysis_data_list)
    
    # Create figure: 1 row, 3 columns; share y-axis.
    fig, axs = plt.subplots(nrows=1, ncols=len_data, figsize=(8, 4), sharey=True)
    if len_data == 1:
        axs = [axs]
    
    # Plot each error analysis data as a bar plot.
    for i, (data, ax) in enumerate(zip(error_analysis_data_list, axs)):
        # Use the original bar-plot function; you can pass middle=True to get the y-label if desired.
        plot_error_analysis_bar(data, ax, middle=True if i == 0 else False)
        # Remove the default x-label and x-tick labels.
        ax.set_xlabel('')
        ax.set_xticks([])
        # Add the model name as text underneath the bar plot.
        ax.text(0.5, -0.15, titles[i],
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=16, fontweight='bold')
    
    # Gather legend handles and labels from all subplots.
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    # Place the combined legend at the top center.
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])