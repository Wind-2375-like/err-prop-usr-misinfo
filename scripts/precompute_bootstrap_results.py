import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.data_loader import load_dataset_results, load_api_key
from utils.evaluator.cot_evaluator import CoTEvaluator
from utils.analysis.metrics import perturbation_ratio_given_correct
from utils.analysis.bootstrapping import bootstrap, bootstrap_with_ratios
from utils.analysis.data_processing import assign_confidence_bins

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on the given dataset')
    parser.add_argument('--sample_size', type=int, default=400, help='Number of samples to generate responses for')
    parser.add_argument('--api_config_file_path', type=str, default="./api_key/config.json", help='API config file path')
    parser.add_argument('--mode', type=str, default="premise", help='Mode to run the experiments')
    parser.add_argument('--base_output_path', type=str, default="./exp_results/eval", help='Base output path')
    return parser.parse_args()

############################################
# Configuration
############################################

args = parse_args()
datasets = ["mix"]
models = [
    "Llama-3.2-90B-Vision-Instruct-Turbo",
    "Qwen2-72B-Instruct",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x22B-Instruct-v0.1",
    "gpt-4o-mini",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.2-11B-Vision-Instruct",
]
mode = "premise"
sample_size = 400
columns = ["prfx_q", "prfx_pert_prfx_q", "prfx_pert_prfx_q_both", "prfx_pert_prfx_q_2step"]
base_output_path = args.base_output_path
config_key = args.api_config_file_path
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)

api_key = load_api_key(config_key)
cot_evaluator = CoTEvaluator(api_key=api_key)

############################################
# Step 1: Evaluate CoT and Store Results
############################################

# overall_results will store {model: {dataset: df_eval_results}}
overall_results = {}

print("Evaluating CoT outputs...")
for model_name in tqdm(models, desc="Models"):
    overall_results[model_name] = {}
    for dataset_name in tqdm(datasets, desc=f"{model_name} Datasets", leave=False):
        try:
            df_sample = load_dataset_results(dataset_name, model_name, sample_size, mode)
        except:
            continue

        df_sample["correct_answer"] = df_sample["correct_answer"].astype(str)

        df_eval_results = pd.DataFrame(columns=columns)
        for idx, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0], desc=f"Evaluating {model_name}-{dataset_name}", leave=False):
            evaluation_results = {}
            for column in columns:
                if column == "prfx_q":
                    evaluation_results[column] = cot_evaluator.evaluate_cot_list(
                        row[column], row["correct_answer"]
                    )
                else:
                    evaluation_results[column] = {}
                    for role in ["human", "self"]:
                        if role in row[column]:
                            evaluation_results[column][role] = cot_evaluator.evaluate_cot_list(
                                row[column][role],
                                row["correct_answer"]
                            )
                        else:
                            evaluation_results[column][role] = {}
            df_eval_results = pd.concat([df_eval_results, pd.DataFrame([evaluation_results], index=[idx])])

        df_eval_results.reset_index(drop=True, inplace=True)
        overall_results[model_name][dataset_name] = df_eval_results

with open(os.path.join(base_output_path, "raw_eval_results.pkl"), "wb") as f:
    pickle.dump(overall_results, f)

############################################
# Step 2: Compute Bootstrapping Results (Radar)
############################################
# Compute ratio lists, mean, lb, ub for each model/dataset/source for radar and main table data.
print("Computing bootstrapping results for radar/table...")
df_results = {}
for model_name in tqdm(models, desc="Radar Bootstrapping - Models"):
    if model_name not in overall_results:
        continue
    df_results[model_name] = {}
    for source in tqdm(["human", "self"], desc=f"{model_name} Sources", leave=False):
        mean_df = pd.DataFrame(index=datasets, columns=columns)
        margin_df = pd.DataFrame(index=datasets, columns=columns)
        ratio_lists = {col: {} for col in columns}

        for column in tqdm(columns, desc=f"{model_name}-{source} Columns", leave=False):
            for dataset_name in tqdm(datasets, desc=f"{model_name}-{source}-{column} Datasets", leave=False):
                if dataset_name not in overall_results[model_name]:
                    mean_df.loc[dataset_name, column] = np.nan
                    margin_df.loc[dataset_name, column] = (np.nan, np.nan)
                    ratio_lists[column][dataset_name] = []
                    continue

                df_data = overall_results[model_name][dataset_name].reset_index(drop=True)
                ratios = bootstrap_with_ratios(
                    df_data,
                    lambda x: perturbation_ratio_given_correct(
                        x,
                        cij="prfx_q",
                        pij=column,
                        evaluation_type="overall_correct",
                        perturbation_role=source if column != "prfx_q" else None
                    ),
                    n=1000,
                )
                mean_val = np.mean(ratios)
                lb_val = np.percentile(ratios, 2.5)
                ub_val = np.percentile(ratios, 97.5)

                mean_df.loc[dataset_name, column] = mean_val
                margin_df.loc[dataset_name, column] = (lb_val, ub_val)
                ratio_lists[column][dataset_name] = ratios

        df_results[model_name][source] = {
            "mean": mean_df,
            "margin": margin_df,
            "ratio_lists": ratio_lists
        }

with open(os.path.join(base_output_path, "results.pkl"), "wb") as f:
    pickle.dump(df_results, f)

############################################
# Step 3: Compute Data for Line Plot (Confidence Bins)
############################################
print("Computing bootstrapping results for line plot...")
line_plot_data = {}
for model_name in tqdm(models, desc="Line Plot - Models"):
    if model_name not in line_plot_data:
        line_plot_data[model_name] = {}
    if model_name not in overall_results or 'mix' not in overall_results[model_name]:
        continue
    df_mix = overall_results[model_name]['mix'].copy()
    df_mix["accuracy"] = df_mix.apply(
        lambda x: sum([v["overall_correct"] for v in x["prfx_q"].values()]) / len(x["prfx_q"]),
        axis=1
    )
    df_binned = assign_confidence_bins(df_mix, score_column="accuracy", n_bins=5)

    # For each column, compute bootstrap in each bin
    for source in ["human", "self"]:
        bin_results = {}
        for column in tqdm(columns, desc=f"{model_name} Line Columns", leave=False):
            bin_list = []
            for cl in tqdm(range(1,6), desc=f"{model_name}-{column} Bins", leave=False):
                df_bin = df_binned[df_binned["confidence_level"] == cl].reset_index(drop=True)
                if len(df_bin) == 0:
                    bin_list.append({
                        "confidence_level": cl,
                        "mean": np.nan,
                        "lb": np.nan,
                        "ub": np.nan,
                        "ratios": []
                    })
                    continue
                ratios = bootstrap_with_ratios(
                    df_bin,
                    lambda x: perturbation_ratio_given_correct(
                        x, cij="prfx_q", pij=column, evaluation_type="overall_correct", perturbation_role=source
                    ),
                    n=1000
                )
                mean_val = np.mean(ratios)
                lb_val = np.percentile(ratios, 2.5)
                ub_val = np.percentile(ratios, 97.5)
                bin_list.append({
                    "confidence_level": cl,
                    "mean": mean_val,
                    "lb": lb_val,
                    "ub": ub_val,
                    "ratios": ratios
                })
            bin_results[column] = bin_list
        line_plot_data[model_name][source] = bin_results

with open(os.path.join(base_output_path, "line_plot_data.pkl"), "wb") as f:
    pickle.dump(line_plot_data, f)

############################################
# Step 5: Compute Data for Error Analysis
############################################
# We'll do a similar approach as the previous snippet, just precompute data and save it. No plotting here.

error_analysis_data = {}
# Example for "Llama-3.2-1B-Instruct" and "mathqa", adjust as needed.
# If files not found, we skip.
for error_model in ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.2-11B-Vision-Instruct"]:
    error_correction_file = f"./exp_results/mix/test_400_perturbed_premise_{error_model}_correction.jsonl"
    error_base_file = f"./exp_results/mix/test_400_perturbed_premise_{error_model}.jsonl"

    if os.path.exists(error_correction_file) and os.path.exists(error_base_file):
        df_sample = pd.read_json(error_correction_file, lines=True)
        df_temp = pd.read_json(error_base_file, lines=True)
        df_temp = df_temp.iloc[:df_sample.shape[0]]
        df_sample["correct_answer"] = df_sample["correct_answer"].astype(str)
        df_temp["correct_answer"] = df_temp["correct_answer"].astype(str)
        df_sample[["prfx_pert_prfx_q_both", "prfx_pert_prfx_q_2step"]] = df_temp[["prfx_pert_prfx_q_both", "prfx_pert_prfx_q_2step"]]

        columns_err = [
            "prfx_q", "prfx_pert_prfx_q", "prfx_pert_prfx_q_pos",
            "prfx_pert_prfx_q_point_out_only_good", "prfx_pert_prfx_q_point_out_only_bad",
            "prfx_pert_prfx_q_corr_bad", "prfx_pert_prfx_q_both", "prfx_pert_prfx_q_2step"
        ]

        df_eval_err = pd.DataFrame(columns=columns_err)
        print("Evaluating error analysis cases...")
        for idx, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0], desc="Error Analysis Evaluation"):
            evaluation_results = {}
            for column in columns_err:
                if "pert" not in column:
                    evaluation_results[column] = cot_evaluator.evaluate_cot_list(
                        row[column],
                        row["correct_answer"]
                    )
                elif "pos" not in column:
                    evaluation_results[column] = {}
                    evaluation_results[column]["human"] = cot_evaluator.evaluate_cot_list(
                        row[column]["human"],
                        row["correct_answer"],
                    )
                else:
                    evaluation_results[column] = {}
                    if "human" in row[column]:
                        evaluation_results[column]["human"] = {}
                        for k, v in row[column]["human"].items():
                            evaluation_results[column]["human"][k] = cot_evaluator.evaluate_cot_list(
                                v, row["correct_answer"]
                            )
            df_eval_err = pd.concat([df_eval_err, pd.DataFrame([evaluation_results], index=[idx])])

        df_eval_err.reset_index(drop=True, inplace=True)

        # Compute metrics needed for error analysis plot:
        # baseline, perturb_baseline, corr_prompt, corr_step0, mean_ratio_pbq, mean_ratio_pgq, mean_ratio_bq, and positional data

        def perturbation_ratio_given_correct_numeric(df, cij='ci', pij='pi'):
            ci = df[cij].values.astype(int)
            pi = df[pij].values.astype(int)
            count_nominator = np.sum(ci * pi)
            count_denominator = np.sum(ci)
            return count_nominator / count_denominator if count_denominator != 0 else np.nan

        print("Bootstrapping error analysis metrics...")
        baseline_mean, baseline_lb, baseline_ub = df_results[error_model]["human"]["mean"]["prfx_q"].values[-1], df_results[error_model]["human"]["margin"]["prfx_q"].values[-1][0], df_results[error_model]["human"]["margin"]["prfx_q"].values[-1][1]
        perturb_baseline_mean, perturb_baseline_lb, perturb_baseline_ub = df_results[error_model]["human"]["mean"]["prfx_pert_prfx_q"].values[-1], df_results[error_model]["human"]["margin"]["prfx_pert_prfx_q"].values[-1][0], df_results[error_model]["human"]["margin"]["prfx_pert_prfx_q"].values[-1][1]
        corr_prompt_mean, corr_prompt_lb, corr_prompt_ub = df_results[error_model]["human"]["mean"]["prfx_pert_prfx_q_2step"].values[-1], df_results[error_model]["human"]["margin"]["prfx_pert_prfx_q_2step"].values[-1][0], df_results[error_model]["human"]["margin"]["prfx_pert_prfx_q_2step"].values[-1][1]
        corr_step0_mean, corr_step0_lb, corr_step0_ub = df_results[error_model]["human"]["mean"]["prfx_pert_prfx_q_both"].values[-1], df_results[error_model]["human"]["margin"]["prfx_pert_prfx_q_both"].values[-1][0], df_results[error_model]["human"]["margin"]["prfx_pert_prfx_q_both"].values[-1][1]
        mean_ratio_pbq, ci_ratio_pbq_lb, ci_ratio_pbq_ub = bootstrap(
            df_eval_err,
            lambda x: perturbation_ratio_given_correct(x, cij='prfx_q', pij='prfx_pert_prfx_q_point_out_only_bad', evaluation_type='overall_correct', perturbation_role='human')
        )
        mean_ratio_pgq, ci_ratio_pgq_lb, ci_ratio_pgq_ub = bootstrap(
            df_eval_err,
            lambda x: perturbation_ratio_given_correct(x, cij='prfx_q', pij='prfx_pert_prfx_q_point_out_only_good', evaluation_type='overall_correct', perturbation_role='human')
        )
        mean_ratio_bq, ci_ratio_bq_lb, ci_ratio_bq_ub = bootstrap(
            df_eval_err,
            lambda x: perturbation_ratio_given_correct(x, cij='prfx_q', pij='prfx_pert_prfx_q_corr_bad', evaluation_type='overall_correct', perturbation_role='human')
        )

        # Positional analysis:
        data_list = []
        for idx, row in df_eval_err.iterrows():
            ci_dict = row['prfx_q']
            ci_values = {k: int(v['overall_correct']) for k, v in ci_dict.items()}
            if 'prfx_pert_prfx_q_pos' in row and 'human' in row['prfx_pert_prfx_q_pos']:
                pert_dict = row['prfx_pert_prfx_q_pos']['human']
                lte_s_keys = [k for k in pert_dict.keys() if k.startswith('lte_s_')]
                lte_s_indices = []
                for k in lte_s_keys:
                    if k != 'lte_s_prompt':
                        idx_str = k.split('_')[2]
                        try:
                            idx_num = int(idx_str)
                            lte_s_indices.append(idx_num)
                        except ValueError:
                            pass
                num_steps = max(lte_s_indices) if lte_s_indices else 1
                for key in lte_s_keys:
                    if key == 'lte_s_prompt':
                        percentage = -1
                    else:
                        i = int(key.split('_')[2])
                        percentage = (i / num_steps)*100 if num_steps>0 else 0
                        percentage = np.clip(percentage,0,100)
                    data_pi_dict = pert_dict[key]
                    for c_k in ci_values.keys():
                        if c_k in data_pi_dict:
                            ci = ci_values[c_k]
                            pi = int(data_pi_dict[c_k]['overall_correct'])
                            data_list.append({
                                'ci': ci,
                                'pi': pi,
                                'percentage': percentage
                            })

        df_data = pd.DataFrame(data_list)
        # -1, 0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100
        bins = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100+1e-6]
        bin_labels = ['prompt','0%-10%','10%-20%','20%-30%','30%-40%','40%-50%',
                    '50%-60%','60%-70%','70%-80%','80%-90%','90%-100%']
        df_data['position'] = pd.cut(df_data['percentage'].apply(lambda x: -1 if x==-1 else x),
                                    bins=bins, labels=bin_labels, right=False, include_lowest=True)
        grouped = df_data.groupby('position')
        results = []
        print("Bootstrapping positional error analysis...")
        for position, group in tqdm(grouped, desc="Error Analysis Positions"):
            mean_r, lb, ub = bootstrap(
                group,
                lambda x: perturbation_ratio_given_correct_numeric(x, cij='ci', pij='pi')
            )
            results.append({
                'position': position,
                'mean_ratio': mean_r,
                'ci': (lb, ub),
                'label': 'Perturbation Positions'
            })

        additional_results = [
            {
                'position': '0%',
                'mean_ratio': mean_ratio_bq,
                'ci': (ci_ratio_bq_lb, ci_ratio_bq_ub),
                'label': 'Error Detection and Correction (Bad Quality)'
            },
            {
                'position': '0%',
                'mean_ratio': mean_ratio_pgq,
                'ci': (ci_ratio_pgq_lb, ci_ratio_pgq_ub),
                'label': 'Error Detection Only (Perfect Quality)'
            },
            {
                'position': '0%',
                'mean_ratio': mean_ratio_pbq,
                'ci': (ci_ratio_pbq_lb, ci_ratio_pbq_ub),
                'label': 'Error Detection Only (Bad Quality)'
            },
            {
                'position': 'prompt',
                'mean_ratio': corr_prompt_mean,
                'ci': (corr_prompt_lb, corr_prompt_ub),
                'label': 'Error Detection and Correction (Baseline Performance)'
            },
            {
                'position': '0%',
                'mean_ratio': corr_step0_mean,
                'ci': (corr_step0_lb, corr_step0_ub),
                'label': 'Error Detection and Correction (Baseline Performance)'
            }
        ]

        error_analysis_data = {
            "baseline": (baseline_mean, baseline_lb, baseline_ub),
            "perturb_baseline": (perturb_baseline_mean, perturb_baseline_lb, perturb_baseline_ub),
            "corr_prompt": (corr_prompt_mean, corr_prompt_lb, corr_prompt_ub),
            "corr_step0": (corr_step0_mean, corr_step0_lb, corr_step0_ub),
            "ratio_pbq": (mean_ratio_pbq, ci_ratio_pbq_lb, ci_ratio_pbq_ub),
            "ratio_pgq": (mean_ratio_pgq, ci_ratio_pgq_lb, ci_ratio_pgq_ub),
            "ratio_bq": (mean_ratio_bq, ci_ratio_bq_lb, ci_ratio_bq_ub),
            "main_results": results,
            "additional_results": additional_results,
            "position_order": bin_labels
        }

        with open(os.path.join(base_output_path, f"error_analysis_data_{error_model}.pkl"), "wb") as f:
            pickle.dump(error_analysis_data, f)
    else:
        # If files not found, save empty dict
        with open(os.path.join(base_output_path, f"error_analysis_data_{error_model}.pkl"), "wb") as f:
            pickle.dump({}, f)

print("Precomputation done. All results saved in ../exp_results/eval.")
