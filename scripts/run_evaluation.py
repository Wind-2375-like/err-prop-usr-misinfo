import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pickle
import pandas as pd
from tqdm import tqdm
from utils.evaluator.cot_evaluator import CoTEvaluator
import argparse
from utils.data_loader import load_dataset_results, load_api_key

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="premise")
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument('--api_config_file_path', type=str, default="./api_key/config.json", help='API config file path')
    parser.add_argument('--dataset_name', type=str, default="mix", help='Dataset name')
    parser.add_argument('--model_names', nargs='+', default=["meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "Qwen/Qwen2-72B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x22B-Instruct-v0.1", "gpt-4o-mini", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-11B-Vision-Instruct"], help='Model names')
    parser.add_argument('--columns', nargs='+', default=["prfx_pert_prfx_q", "prfx_pert_prfx_q_both"], help='Columns')
    parser.add_argument('--output_path', type=str, default="./exp_results/eval/test_400_perturbed_premise_evaluated.pkl", help='Output path')
    return parser.parse_args()
    
def aggregate_sample(args, openai_api_key):
    # Instead of starting with a DataFrame, we'll start with an empty list
    rows_to_add = []
    cot_evaluator = initialize_cot_evaluator(openai_api_key)

    for model_name in tqdm(args.model_names, desc="Loading samples"):
        # Load the dataset results (make sure load_dataset_results handles caching or is fast)
        df_sample = load_dataset_results(args.dataset_name, model_name.split('/')[-1], args.sample_size, args.mode)
        df_sample["correct_answer"] = df_sample["correct_answer"].astype(str)
        # Using itertuples or iterrows can be considered; itertuples is slightly faster
        for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc=f"Processing {model_name}"):
            for source in ["human"]:
                for i in range(5):
                    try:
                        original_output = row["prfx_q"][f"c_{i}"]
                        output = row["prfx_pert_prfx_q"]["human" if source == "human" else "self"][f"c_{i}"]
                        corrected_output = row["prfx_pert_prfx_q_both"]["human" if source == "human" else "self"][f"c_{i}"]
                        try:
                            overall_correct_original = cot_evaluator.answer_verifier(original_output[-1], row["correct_answer"])
                        except:
                            continue
                        # Append to rows_to_add if the original answer is correct
                        if overall_correct_original:
                            try:
                                overall_correct = cot_evaluator.answer_verifier(output[-1], row["correct_answer"])
                                point_out_correct = cot_evaluator.answer_verifier(corrected_output[-1], row["correct_answer"])
                                rows_to_add.append({
                                    "question": row["question"],
                                    "correct_answer": row["correct_answer"],
                                    "premise": row["premise"][source],
                                    "perturbed_premise": row["perturbed_premise"][source],
                                    "overall_correct": overall_correct,
                                    "point_out_correct": point_out_correct,
                                    "output": output,
                                    "point_out_output": corrected_output,
                                    "model": model_name,
                                    "dataset": args.dataset_name,
                                })
                            except:
                                continue
                    except KeyError as e:
                        raise

    # Convert the accumulated rows into a DataFrame once
    df_aggregated = pd.DataFrame(rows_to_add)
    
    def filter_out_truncated_answer(row):
        return 'answer' in row['output'][-1] and 'answer' in row['point_out_output'][-1]

    # Filter and sample
    df_aggregated = df_aggregated[df_aggregated["output"].apply(lambda x: x != []) & df_aggregated["point_out_output"].apply(lambda x: x != [])]
    df_aggregated = df_aggregated[df_aggregated.apply(filter_out_truncated_answer, axis=1)]

    # Sample 400 from each column
    df_sample = df_aggregated.sample(n=args.sample_size, random_state=42)
    return df_sample

def initialize_cot_evaluator(openai_api_key):
    cot_evaluator = CoTEvaluator(api_key=openai_api_key, point_out_model_name='gpt-4o')
    return cot_evaluator

def perform_evaluation(df_sample, cot_evaluator, args):
    df_evaluated = df_sample.copy()
    df_evaluated["detection"] = df_evaluated.apply(lambda x: None, axis=1)
    df_evaluated["correction"] = df_evaluated.apply(lambda x: None, axis=1)
    df_evaluated["perturbation"] = df_evaluated.apply(lambda x: None, axis=1)
    df_evaluated["detection_prompting"] = df_evaluated.apply(lambda x: None, axis=1)
    df_evaluated["correction_prompting"] = df_evaluated.apply(lambda x: None, axis=1)
    df_evaluated["perturbation_prompting"] = df_evaluated.apply(lambda x: None, axis=1)

    # Use a tqdm progress bar
    with tqdm(total=len(df_sample), desc="Evaluating samples") as pbar:
        for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            detection_label, detection_explanations = cot_evaluator.overall_point_out_error_verifier(row["question"], row["output"], row["perturbed_premise"])
            if detection_label:
                detection_positions = cot_evaluator.point_out_position_verifier(row["question"], row["output"], row["perturbed_premise"], detection_explanations)
                correction_label, correction_explanations = cot_evaluator.overall_success_correction_verifier(row["question"], row["output"], row["premise"], row["perturbed_premise"])
            else:
                detection_positions = []
                correction_label, correction_explanations = False, detection_explanations
            perturbation_label, perturbation_explanations = cot_evaluator.overall_perturbation_verifier(row["question"], row["output"], row["premise"], row["perturbed_premise"])
            
            detection_prompting_label, detection_prompting_explanations = cot_evaluator.overall_point_out_error_verifier(row["question"], row["point_out_output"], row["perturbed_premise"])
            if detection_prompting_label:
                correction_prompting_label, correction_prompting_explanations = cot_evaluator.overall_success_correction_verifier(row["question"], row["point_out_output"], row["premise"], row["perturbed_premise"])
            else:
                correction_prompting_label, correction_prompting_explanations = False, detection_prompting_explanations
            perturbation_prompting_label, perturbation_prompting_explanations = cot_evaluator.overall_perturbation_verifier(row["question"], row["point_out_output"], row["premise"], row["perturbed_premise"])
                
            df_evaluated.at[idx, "detection"] = {"label": detection_label, "explanations": detection_explanations, "steps": detection_positions}
            df_evaluated.at[idx, "correction"] = {"label": correction_label, "explanations": correction_explanations, "steps": detection_positions}
            df_evaluated.at[idx, "perturbation"] = {"label": perturbation_label, "explanations": perturbation_explanations}
            df_evaluated.at[idx, "detection_prompting"] = {"label": detection_prompting_label, "explanations": detection_prompting_explanations}
            df_evaluated.at[idx, "correction_prompting"] = {"label": correction_prompting_label, "explanations": correction_prompting_explanations}
            df_evaluated.at[idx, "perturbation_prompting"] = {"label": perturbation_prompting_label, "explanations": perturbation_prompting_explanations}
            
            # Get usage from cot_evaluator and update tqdm
            try:
                usage = cot_evaluator.get_usage()["gpt-4o"]
            except:
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            pbar.set_postfix({
                "Prompt tokens": usage['prompt_tokens'], 
                "Completion tokens": usage['completion_tokens'], 
                "Total tokens": usage['total_tokens']
            })
            pbar.update(1)
            
            # Save the evaluated DataFrame
            with open(args.output_path, "wb") as f:
                pickle.dump(df_evaluated, f)

def report_usage(cot_evaluator):
    # Report token usage
    for model, usage in cot_evaluator.get_usage().items():
        print(f"Model: {model}")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print()
    return

def main():
    args = parse_args()
    openai_api_key = load_api_key(args.api_config_file_path)
    df_sample = aggregate_sample(args, openai_api_key)
    cot_evaluator = initialize_cot_evaluator(openai_api_key)
    perform_evaluation(df_sample, cot_evaluator, args)
    report_usage(cot_evaluator)

if __name__ == "__main__":
    main()