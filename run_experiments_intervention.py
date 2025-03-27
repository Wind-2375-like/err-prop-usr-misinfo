import pandas as pd
from tqdm import tqdm
import json
from utils.generator.prediction_generator import CounterfactualPremisePredictionGenerator
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on the given dataset')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name to use for generating responses')
    parser.add_argument('--dataset_name', type=str, default="math", help='Dataset name to use for generating responses')
    parser.add_argument('--sample_size', type=int, default=400, help='Number of samples to generate responses for')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature value for generating responses')
    parser.add_argument('--top_p', type=float, default=0.7, help='Top p value for generating responses')
    parser.add_argument('--top_k', type=int, default=50, help='Top k value for generating responses')
    parser.add_argument('--number_of_outputs', type=int, default=5, help='Number of outputs to generate')
    parser.add_argument('--api_config_file_path', type=str, default="./api_key/config.json", help='API config file path')
    parser.add_argument('--processed_data_path', type=str, default="./pcd_data", help='Processed data path')
    parser.add_argument('--output_file_path', type=str, default="./exp_results", help='Output file path')
    parser.add_argument('--mode', type=str, default="premise", help='Mode to run the experiments')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for the dataset')
    return parser.parse_args()


def main():
    args = parse_args()
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    number_of_outputs = args.number_of_outputs
    
    # Load the api key
    with open (f"{args.api_config_file_path}", "r") as f:
        config = json.load(f)
        args.openai_api_key = config["openai_api_key"]
        args.togetherai_api_key = config["togetherai_api_key"]
    
    # Load the dataset, run the experiments, and save the results one by one
    with open(f"{args.processed_data_path}/{args.dataset_name}/test_{args.sample_size}_perturbed.jsonl", "r") as f:
        data = [json.loads(row) for row in f]
    start = args.start_idx
    data = data[start:]
    df_sample = data.copy()
    df_premise = pd.read_json(f"{args.output_file_path}/{args.dataset_name}/test_{args.sample_size}_perturbed_premise_{args.model_name[args.model_name.find('/')+1:]}.jsonl", lines=True)

    # Premise perturbation
    if args.mode == "premise":
        prediction_generator = CounterfactualPremisePredictionGenerator(
            model_name=args.model_name,
            api_key=args.togetherai_api_key if "gpt" not in args.model_name else args.openai_api_key,
            local=True,
        )
        for idx, row in tqdm(enumerate(data), total=len(data)):
            row["prfx_pert_prfx_q"] = df_premise.iloc[idx]["prfx_pert_prfx_q"]
            # n should be 5
            prfx_pert_prfx_q_point_out_only_good = {
                "human": prediction_generator.run_premise_experiment(
                    row,
                    model_name="human",
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="step0",
                    point_out_only=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )["lte_s_0"],
                "self": prediction_generator.run_premise_experiment(
                    row,
                    model_name=args.model_name,
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="step0",
                    point_out_only=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )["lte_s_0"]
            }
            prfx_pert_prfx_q_point_out_only_bad = {
                "human": prediction_generator.run_premise_experiment(
                    row,
                    model_name="human",
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="step0",
                    point_out_only=True,
                    bad_quality=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )["lte_s_0"],
                "self": prediction_generator.run_premise_experiment(
                    row,
                    model_name=args.model_name,
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="step0",
                    point_out_only=True,
                    bad_quality=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )["lte_s_0"]
            }
            prfx_pert_prfx_q_corr_bad = {
                "human": prediction_generator.run_premise_experiment(
                    row,
                    model_name="human",
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="step0",
                    bad_quality=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )["lte_s_0"],
                "self": prediction_generator.run_premise_experiment(
                    row,
                    model_name=args.model_name,
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="step0",
                    bad_quality=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )["lte_s_0"]
            }
            prfx_pert_prfx_q_corr_user = {
                "human": prediction_generator.run_premise_experiment(
                    row,
                    model_name="human",
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="user",
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                ),
                "self": prediction_generator.run_premise_experiment(
                    row,
                    model_name=args.model_name,
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="user",
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )
            }
            prfx_pert_prfx_q_corr_pos = {
                "human": prediction_generator.run_premise_experiment(
                    row,
                    model_name="human",
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="cot",
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                ),
                "self": prediction_generator.run_premise_experiment(
                    row,
                    model_name=args.model_name,
                    perturb=True,
                    input_correction=True,
                    warning=True,
                    pos="cot",
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=number_of_outputs
                )
            }
            prfx_pert_prfx_q_corr_pos["human"]["lte_s_prompt"] = prfx_pert_prfx_q_corr_user["human"]
            prfx_pert_prfx_q_corr_pos["self"]["lte_s_prompt"] = prfx_pert_prfx_q_corr_user["self"]
            
            df_sample[idx]["prfx_q"] = df_premise.iloc[idx]["prfx_q"]
            df_sample[idx]["prfx_pert_prfx_q"] = df_premise.iloc[idx]["prfx_pert_prfx_q"]
            df_sample[idx]["prfx_pert_prfx_q_pos"] = prfx_pert_prfx_q_corr_pos
            df_sample[idx]["prfx_pert_prfx_q_point_out_only_good"] = prfx_pert_prfx_q_point_out_only_good
            df_sample[idx]["prfx_pert_prfx_q_point_out_only_bad"] = prfx_pert_prfx_q_point_out_only_bad
            df_sample[idx]["prfx_pert_prfx_q_corr_bad"] = prfx_pert_prfx_q_corr_bad
            
            with open(f"{args.output_file_path}/{args.dataset_name}/test_{args.sample_size}_perturbed_{args.mode}_{args.model_name[args.model_name.find('/')+1:]}_correction.jsonl", "a") as f:
                f.write(json.dumps(df_sample[idx]) + "\n")
    
if __name__ == "__main__":
    main()