import pandas as pd
from tqdm import tqdm
import json
from utils.generator.prediction_generator import PremisePredictionGenerator
import argparse

def get_model_name(model_name, finetuned_model_name):
    if model_name == "gpt-4o-mini-sft":
        return finetuned_model_name    
    else:
        return model_name  


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on the given dataset')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini-sft", help='Fine-tuned model name to use for generating responses')
    parser.add_argument('--dataset_name', type=str, default="mix", help='Dataset name to use for generating responses')
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
        args.actual_finetuned_model_name = config[args.model_name]
    
    # Load the dataset, run the experiments, and save the results one by one
    with open(f"{args.processed_data_path}/{args.dataset_name}/test_{args.sample_size}_perturbed.jsonl", "r") as f:
        data = [json.loads(row) for row in f]
    start = args.start_idx
    data = data[start:]
    df_sample = data.copy()
    
    with open(f"{args.output_file_path}/{args.dataset_name}/test_{args.sample_size}_perturbed_{args.mode}_gpt-4o-mini.jsonl", "r") as f:
        base_data = [json.loads(row) for row in f]
    base_data = base_data[start:]
    df_sample_base = base_data.copy()
    
    # Premise perturbation
    if args.mode == "premise":
        base_prediction_generator = PremisePredictionGenerator(
            model_name="gpt-4o-mini",
            api_key=args.togetherai_api_key if "gpt" not in args.model_name else args.openai_api_key
        )
        prediction_generator = PremisePredictionGenerator(
            model_name=get_model_name(args.model_name, args.actual_finetuned_model_name),
            api_key=args.togetherai_api_key if "gpt" not in args.model_name else args.openai_api_key
        )
        for idx, row in tqdm(enumerate(data), total=len(data)):
            df_sample[idx]["base_original"] = df_sample_base[idx]["prfx_q"]
            df_sample[idx]["base_misinformed"] = {
                "human": df_sample_base[idx]["prfx_pert_prfx_q"]["human"]
            }
            df_sample[idx]["inst_original"] = base_prediction_generator.run_premise_experiment(
                row,
                warning=True,
                model_name="self",
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                n=number_of_outputs,
            )
            df_sample[idx]["inst_misinformed"] = {
                "human": df_sample_base[idx]["prfx_pert_prfx_q_both"]["human"]
            }
            df_sample[idx]["ft_original"] = prediction_generator.run_premise_experiment(
                row,
                model_name="self",
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                n=number_of_outputs,
            )
            df_sample[idx]["ft_misinformed"] = {
                "human": prediction_generator.run_premise_experiment(
                    row,
                    model_name="human",
                    perturb=True,
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p,
                    n=number_of_outputs,
                )
            }
            df_sample[idx]["inst_ft_original"] = prediction_generator.run_premise_experiment(
                row,
                warning=True,
                model_name="self",
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                n=number_of_outputs,
            )
            df_sample[idx]["inst_ft_misinformed"] = {
                "human": prediction_generator.run_premise_experiment(
                    row,
                    model_name="human",
                    perturb=True,
                    warning=True,
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p,
                    n=number_of_outputs,
                )
            }
            
            with open(f"{args.output_file_path}/{args.dataset_name}/test_{args.sample_size}_perturbed_{args.mode}_{args.model_name[args.model_name.find('/')+1:]}.jsonl", "a") as f:
                f.write(json.dumps(df_sample[idx]) + "\n")

    # Print total usage
    for model_name, usage in prediction_generator.get_usage().items():
        print(f"Model: {model_name}")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print()
    
if __name__ == "__main__":
    main()