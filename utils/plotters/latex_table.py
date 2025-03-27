def print_latex_table(df_results):
    import math

    model_name_map = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "LlaMAV-3.2-90B",
        "Qwen2-72B-Instruct": "Qwen-2-72B",
        "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8×7B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8×22B",
        "gpt-4o-mini": "GPT-4o-mini",
        "Llama-3.2-1B-Instruct": "LlaMA-3.2-1B",
        "Llama-3.2-3B-Instruct": "LlaMA-3.2-3B",
        "Llama-3.2-11B-Vision-Instruct": "LlaMAV-3.2-11B",
    }
    models_order = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct-Turbo",
        "Qwen2-72B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x22B-Instruct-v0.1",
        "gpt-4o-mini"
    ]

    # Conditions in the exact order as in your snippet
    conditions = [
        ("prfx_q", r"\textit{Accuracy upon re-prompting without misinformation: $\mathbb{E}_{q \in Q_c}[A_{q}^{(2)}]$}"),
        ("prfx_pert_prfx_q", r"\textit{Accuracy under misinformation: $\mathbb{E}_{q \in Q_c}[A_{q_{m}}^{(2)}]$}"),
        ("prfx_pert_prfx_q_2step", r"\textit{Accuracy under misinformation: $\mathbb{E}_{q \in Q_c}[A_{q_{m}}^{(2)}]$ + Error correction at user prompts}"),
        ("prfx_pert_prfx_q_both", r"\textit{Accuracy under misinformation: $\mathbb{E}_{q \in Q_c}[A_{q_{m}}^{(2)}]$ + Error correction at the first CoT steps}")
    ]

    def format_cell(mean, margin):
        if math.isnan(mean):
            return "N/A"
        else:
            return f"{mean:.2f} \\tiny"+r"{"+f"[{margin[0]:.2f}, {margin[1]:.2f}]"+r"}"

    # Print the LaTeX table header exactly as given
    print(r"\begin{tabular}{p{3.0cm}|p{2.35cm}p{2.35cm}p{2.7cm}p{2.7cm}p{2.25cm}p{2.35cm}p{2.35cm}p{2.25cm}}")
    print(r"\toprule")
    print("        & " + " & ".join([model_name_map[m] for m in models_order]) + r" \\ \midgrayline")

    for i, (col, col_header) in enumerate(conditions):
        print(r"\multicolumn{9}{>{\columncolor{\grayColor}} c}{"+col_header+"}\\\ \midgrayline")
        if col == "prfx_q":
            row_strs = ["Original"]
        else:
            row_strs = ["Out-of-distribution", "In-distribution"]
           
        for row_s in row_strs: 
            row_str = row_s
            for model_key in models_order:
                if row_s == "In-distribution":
                    mean_val = df_results[model_key]["self"]["mean"][col].values[-1]
                    margin_val = df_results[model_key]["self"]["margin"][col].values[-1]
                else:
                    mean_val = df_results[model_key]["human"]["mean"][col].values[-1]
                    margin_val = df_results[model_key]["human"]["margin"][col].values[-1]
                cell_str = format_cell(mean_val, margin_val)
                row_str += " & " + cell_str

            row_str += r" \\"
            print(row_str)

        # After each scenario except the last one, print a midgrayline
        if i < len(conditions)-1:
            print(r"\midgrayline")

    print(r"\bottomrule")
    print(r"\end{tabular}")

def print_flat_latex_table(df_results):
    import math

    model_name_map = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "LlaMAV-3.2-90B",
        "Qwen2-72B-Instruct": "Qwen-2-72B",
        "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8×7B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8×22B",
        "gpt-4o-mini": "GPT-4o-mini",
        "Llama-3.2-1B-Instruct": "LlaMA-3.2-1B",
        "Llama-3.2-3B-Instruct": "LlaMA-3.2-3B",
        "Llama-3.2-11B-Vision-Instruct": "LlaMAV-3.2-11B",
    }
    models_order = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct-Turbo",
        "Qwen2-72B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x22B-Instruct-v0.1",
        "gpt-4o-mini",
    ]
    model_names = [model_name_map[model_key] for model_key in models_order]

    # Conditions in the exact order as in your snippet
    conditions = [
        ("prfx_q", "Original"),
        ("prfx_pert_prfx_q", "+Misinfo."),
    ]

    def format_cell(mean, margin):
        if math.isnan(mean):
            return "N/A"
        else:
            return f"{mean:.2f} \\tiny"+r"{"+f"[{margin[0]:.2f}, {margin[1]:.2f}]"+r"}"

    # Print the LaTeX table header exactly as given
    print(r"\begin{tabular}{p{2.7cm}|p{2.35cm}p{2.35cm}p{2.7cm}p{2.7cm}p{2.25cm}p{2.35cm}p{2.35cm}p{2.25cm}}")
    print(r"\toprule")
    print("        & " + " & ".join(model_names) + r" \\ \midgrayline")

    for _, (col, col_header) in enumerate(conditions):
        # Construct the row
        row_str = col_header

        if col == "prfx_q":
            for model_key in models_order:
                # Retrieve data
                try:
                    out_means = df_results[model_key]["human"]["mean"][col].values[-1]
                    out_margins = df_results[model_key]["human"]["margin"][col].values[-1]
                    cell_str = format_cell(out_means, out_margins)
                except:
                    cell_str = "N/A"

                row_str += " & " + cell_str

            row_str += r" \\"
            print(row_str)

            print(r"\midgrayline")
            print(r"\midgrayline")
        else:
            row_str = col_header + " OOD"
            for model_key in models_order:
                # Retrieve data
                try:
                    out_means = df_results[model_key]["human"]["mean"][col].values[-1]
                    out_margins = df_results[model_key]["human"]["margin"][col].values[-1]
                    cell_str = format_cell(out_means, out_margins)
                except:
                    cell_str = "N/A"

                row_str += " & " + cell_str
                
            row_str += r" \\"
            print(row_str)

            print(r"\midgrayline")

            row_str = "Relative Decrease"
            # Print Relative Decrease
            for model_key in models_order:
                try:
                    out_means_ori = df_results[model_key]["human"]["mean"]["prfx_q"].values[-1]
                    out_means_now = df_results[model_key]["human"]["mean"]["prfx_pert_prfx_q"].values[-1]
                    cell_str = f"{(out_means_ori - out_means_now) / out_means_ori * 100:.2f}\%"
                except:
                    cell_str = "N/A"

                row_str += " & " + cell_str
            
    
            row_str += r" \\"
            print(row_str)
            
            print(r"\midgrayline")
            print(r"\midgrayline")
            
            row_str = col_header + " IND"
            for model_key in models_order:
                # Retrieve data
                try:
                    out_means = df_results[model_key]["self"]["mean"][col].values[-1]
                    out_margins = df_results[model_key]["self"]["margin"][col].values[-1]
                    cell_str = format_cell(out_means, out_margins)
                except:
                    cell_str = "N/A"

                row_str += " & " + cell_str
                
            row_str += r" \\"
            print(row_str)
            
            print(r"\midgrayline")
            
            row_str = "Relative Decrease"
            # Print Relative Decrease
            for model_key in models_order:
                try:
                    out_means_ori = df_results[model_key]["self"]["mean"]["prfx_q"].values[-1]
                    out_means_now = df_results[model_key]["self"]["mean"]["prfx_pert_prfx_q"].values[-1]
                    cell_str = f"{(out_means_ori - out_means_now) / out_means_ori * 100:.2f}\%"
                except:
                    cell_str = "N/A"

                row_str += " & " + cell_str
            
    
            row_str += r" \\"
            print(row_str)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    
def print_pearson_correlation_table(pearson_pairs):
    model_name_map = {
        "Llama-3.2-90B-Vision-Instruct-Turbo": "LlaMAV-3.2-90B",
        "Qwen2-72B-Instruct": "Qwen-2-72B",
        "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8×7B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8×22B",
        "gpt-4o-mini": "GPT-4o-mini",
        "Llama-3.2-1B-Instruct": "LlaMA-3.2-1B",
        "Llama-3.2-3B-Instruct": "LlaMA-3.2-3B",
        "Llama-3.2-11B-Vision-Instruct": "LlaMAV-3.2-11B",
        "overall": "All Models"
    }
    models_order = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct-Turbo",
        "Qwen2-72B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x22B-Instruct-v0.1",
        "gpt-4o-mini",
        "overall"
    ]
    model_names = [model_name_map[model_key] for model_key in models_order]
    
    # Print the LaTeX table header exactly as given
    print(r"\begin{tabular}{p{3.0cm}|p{2.35cm}p{2.35cm}p{2.7cm}p{2.7cm}p{2.25cm}p{2.35cm}p{2.35cm}p{2.25cm}p{2.25cm}}")
    print(r"\toprule")
    print(r"\textbf{Pearson Corr.} & " + " & ".join([model_name_map[m] for m in models_order]) + r" \\ \midgrayline")
    
    from scipy.stats import pearsonr
    xs = []
    ys = []
    pearson = []
    for model_key in models_order[:-1]:
        x = [pair["x"] for pair in pearson_pairs[model_key]]
        y = [pair["y"] for pair in pearson_pairs[model_key]]
        xs += x
        ys += y
        pearson.append(pearsonr(x, y))
    pearson.append(pearsonr(xs, ys))
    
    # Plot the coefficients in the first row then the p-values in the second row
    for i in range(2):
        row_str = "Coefficients" if i == 0 else "P-Values"
        for j, model_key in enumerate(models_order):
            row_str += " & " + f"{pearson[j][i]:.2f}"
        row_str += r" \\"
        print(row_str)
        
        if i == 0:
            print(r"\midgrayline")
            
    print(r"\bottomrule")
    print(r"\end{tabular}")