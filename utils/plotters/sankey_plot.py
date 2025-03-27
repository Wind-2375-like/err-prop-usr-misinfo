import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np

def plot_sankey_data(list_of_names, grouped, wide=False):
    """
    Plots a Sankey diagram from grouped DataFrame with columns: list_of_names, counts.
    Saves the figure as PDF and displays the chart in the Jupyter notebook.

    Also uses gradient on the links from the source color to the target color.
    """

    # Sum counts for Overall->ErrorPoint
    overall_error = (
        grouped.groupby(list_of_names)['counts']
        .sum()
        .reset_index()
    )
    # Sort the first and second column by alphabetical order
    if len(overall_error) == 6:
        # Swap row 2,3 with 0,1
        overall_error.iloc[[0, 1, 2, 3, 4, 5]] = overall_error.iloc[[2, 3, 0, 1, 4, 5]].copy()

    # Calculate the grand total
    grand_total = overall_error['counts'].sum()

    # Create a dictionary to store the total values for each unique node
    node_totals = {}
    for _, row in overall_error.iterrows():
        for name in list_of_names:
            node_totals[row[name]] = node_totals.get(row[name], 0) + row['counts']

    # Build the sankey data
    sankey_data = []
    for _, row in overall_error.iterrows():
        for i in range(len(list_of_names) - 1):
            sankey_data.append({
                "from": f"{row[list_of_names[i]]} {(node_totals[row[list_of_names[i]]] / grand_total * 100):.2f}%",
                "to": f"{row[list_of_names[i+1]]} {(node_totals[row[list_of_names[i+1]]] / grand_total * 100):.2f}%",
                "value": int(row['counts'])
            })

    # Combine duplicate flows
    new_sankey_data = []
    for d in sankey_data:
        found = False
        for nd in new_sankey_data:
            if nd["from"] == d["from"] and nd["to"] == d["to"]:
                nd["value"] += d["value"]
                found = True
                break
        if not found:
            new_sankey_data.append(d)

    # Turn sankey data into JS array
    sankey_data_js = ",\n        ".join([
        f'{{ from: "{d["from"]}", to: "{d["to"]}", value: {d["value"]} }}' for d in new_sankey_data
    ])

    # Build the HTML/JS template
    html_template = f"""<!-- Styles -->
<style>
    #chartdiv {{
        width: {'100%' if wide else '50%'};
        height: 80%;
    }}
</style>

<!-- Resources -->
<script src="https://cdn.amcharts.com/lib/5/index.js"></script>
<script src="https://cdn.amcharts.com/lib/5/flow.js"></script>
<script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>

<!-- Chart code -->
<script>
am5.ready(function () {{
    var root = am5.Root.new("chartdiv");

    // Apply theme
    root.setThemes([am5themes_Animated.new(root)]);

    // Create Sankey series
    var series = root.container.children.push(
        am5flow.Sankey.new(root, {{
            sourceIdField: "from",
            targetIdField: "to",
            valueField: "value",
            nodePadding: 15,
            paddingRight: 50,
            paddingTop: 50
        }})
    );
    
    // Set default label styling
    series.nodes.labels.template.setAll({{
        fontWeight: "bold",
        fontSize: {53 if wide else 33},
        fontFamily: "DejaVu Sans",
        x: am5.p50,
        paddingLeft: 15,
        paddingRight: 15,
    }});
    
    // Dynamically position labels based on node connections
    series.nodes.labels.template.adapters.add("centerX", function (center, target) {{
        if (target.dataItem.get("incomingLinks", []).length === 0) {{
            return am5.p0;
        }} else if (target.dataItem.get("outgoingLinks", []).length === 0) {{
            return am5.p100;
        }}
        return am5.p50;
    }});

    // Set node width
    series.set("nodeWidth", 20);

    //--------------------------------------
    // 1) Helper function to pick node color
    //--------------------------------------
    function getNodeColor(name) {{
        // Lowercase for simpler checks
        var n = name;
        // Light-ish versions (adjust to taste)
        if (n.includes("NF-Corr"))    return am5.color(0xff6666); // light red
        if (n.includes("F-Corr"))      return am5.color(0x66cc66); // light green
        if (n.includes("N-Corr"))    return am5.color(0xffee88); // pale yellow
        if (n.includes("Nonfactual"))    return am5.color(0xff6666); // light red
        if (n.includes("Follow"))      return am5.color(0x88aaff); // pale blue
        if (n.includes("Resist"))    return am5.color(0xffee88); // pale yellow
        if (n.includes("❎"))    return am5.color(0xff6666); // light red
        if (n.includes("✅"))      return am5.color(0x66cc66); // light green
        if (n === "") return am5.color(0xffee88); // pale yellow
        return am5.color(0x88aaff);     // fallback to pale blue
    }}

    //--------------------------------------
    // 2) Color the nodes themselves
    //--------------------------------------
    // Turn off built-in multi-step color set so it won't override
    series.nodes.set("colors", am5.ColorSet.new(root, {{ passOptions: {{}} }}));

    // Make node rectangles visible
    series.nodes.rectangles.template.setAll({{
        fillOpacity: 1
    }});

    // Adapter to pick color by name
    series.nodes.rectangles.template.adapters.add("fill", function(fill, target) {{
        var nodeName = target.dataItem.get("name") || "";
        return getNodeColor(nodeName);
    }});

    // 3) Color the NODES
    // Turn off the default color set steps so we control everything
    series.nodes.set("colors", am5.ColorSet.new(root, {{ passOptions: {{}} }}));

    // Make the node rectangles visible
    series.nodes.rectangles.template.setAll({{
        fillOpacity: 1
    }});

    // Attach adapter to color each node rectangle
    series.nodes.rectangles.template.adapters.add("fill", function(fill, target) {{
        var nodeName = target.dataItem.get("name") || "";
        return getNodeColor(nodeName);
    }});

    // 4) Create gradient flows (LINKS)
    var linkTemplate = series.links.template;
    linkTemplate.setAll({{
        fillOpacity: 0.9,
        strokeOpacity: 0.9,
        strokeWidth: 1
    }});
    linkTemplate.states.create("hover", {{ fillOpacity: 1 }});

    // Key trick: use fillGradient instead of just fill
    // to get a 2-stop gradient from source to target color
    linkTemplate.adapters.add("fillGradient", function(fillGradient, link) {{
        var sourceNode = link.dataItem.get("source");
        var targetNode = link.dataItem.get("target");
        if (!sourceNode || !targetNode) return fillGradient;

        var sourceColor = getNodeColor(sourceNode.get("name") || "");
        var targetColor = getNodeColor(targetNode.get("name") || "");

        // Create a brand-new gradient each time
        var gradient = am5.LinearGradient.new(root, {{
            stops: [
                {{ color: sourceColor }},
                {{ color: targetColor }}
            ],
            rotation: 0 // 0 = left-right, 90 = top-bottom, etc.
        }});

        return gradient;
    }});

    // Also match stroke to that same gradient
    linkTemplate.adapters.add("strokeGradient", function(strokeGradient, link) {{
        return link.get("fillGradient");
    }});

    // 5) Add your sankey data
    series.data.setAll([
        {sankey_data_js}
    ]);

    series.appear(1000, 100);
}});
</script>

<!-- HTML -->
<div id="chartdiv"></div>
"""
    return html_template

def plot_position_distribution(df_sankey):
    detection_positions = []
    detection_prompting_positions = []

    for _, sample in df_sankey.iterrows():
        output_length = len(sample['output'])
        point_out_output_length = len(sample['point_out_output'])
        try:
            detection_positions.extend([i/output_length for i in sample["detection"]["steps"]])
            detection_prompting_positions.extend([i/point_out_output_length for i in sample["detection_prompting"]["steps"]])
        except:
            continue

    # Plot the distribution of kde of detection positions
    mpl.rcParams.update({
        'font.size': 20,        # base font size
        'axes.labelsize': 20,   # x and y labels
        'axes.titlesize': 20,   # subplot title
        'xtick.labelsize': 20,  # x tick labels
        'ytick.labelsize': 20,  # y tick labels
        'legend.fontsize': 20   ,  # legend
    })
    
    bins = np.linspace(0, 1, 11)
    
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    
    ax1.hist(detection_positions, bins=bins, alpha=0.4, color='#ff7f0e', edgecolor='#ff7f0e')
    ax1.set_yscale('log')
    ax1.set_ylim(1, 50)
    ax1.set_ylabel("Frequency")
    
    sns.kdeplot(detection_positions, ax=ax2, bw_adjust=0.4, color='#ff7f0e')
    ax2.set_ylabel("KDE Density")
    ax2.set_ylim(0, 2)
    
    ax1.set_xlabel("Position")
    ax1.set_xlim(0, 1)
    ticks = np.linspace(0, 1, 11)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f"{int(x * 100)}%" for x in ticks])
    
    color_mis = mcolors.to_rgba('#ff7f0e', alpha=0.5)
    handle_mis = Rectangle((0, 0), 1, 1, facecolor=color_mis, edgecolor='#ff7f0e', linewidth=1)
    
    ax1.legend([handle_mis],
               ['Misinformed'],
               loc='upper right',
               frameon=True)
    # ax1.legend(["Misinformed"], loc='upper right', frameon=True)
    ax1.grid(False)
    
    plt.tight_layout()