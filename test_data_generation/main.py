import argparse
import pandas as pd
import json
from tqdm import tqdm
import os
import sys

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import your modules
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.generator.cot_generator import CoTGenerator
from utils.generator.premise_generator import PremiseGenerator
from utils.generator.perturbation_generator import PerturbationGenerator
from utils.evaluator import CoTEvaluator
from test_data_generation.processors import get_processor
from test_data_generation.tools.file_io import read_jsonl_file

def get_data_path(dataset_name):
    # Define possible paths based on where the code might be run
    possible_paths = [
        os.path.join("raw_data", dataset_name, "test.jsonl"),  # Project root path
        os.path.join("..", "raw_data", dataset_name, "test.jsonl"),  # Parent directory
        os.path.join(script_dir, "raw_data", dataset_name, "test.jsonl"),  # Relative to script
        os.path.join(parent_dir, "raw_data", dataset_name, "test.jsonl")  # Relative to project root
    ]
    
    # Check each path to see if it exists
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If no valid path is found, raise an error
    raise FileNotFoundError(
        f"Could not find the file 'test.jsonl' for dataset '{dataset_name}' in any of the expected locations. "
        f"Checked paths: {possible_paths}"
    )


def get_init_df(dataset_name):
    # Read raw data using the dynamically determined path
    data_path = get_data_path(dataset_name)
    
    # Load the data
    raw_df = read_jsonl_file(data_path)
    
    # Get the appropriate processor for the dataset
    processor = get_processor(dataset_name)
    
    # Process the data
    processed_df = processor.process_data(raw_df)
    
    return processed_df

def generate_examples(args):
    for dataset_name in args.dataset_names:
        sample_size = args.sample_size
        count = args.start_count
        df_idx = args.start_idx
        system_content_premise = "You are given a question. Generate only LaTeX formulas for the question without ever answering the question or revealing the answer. Each formula should be wrapped between single dollar signs and separated by semicolons. The variables should be either from the question or wrapped in $\\text{...}$.\n\nExample:\n\nQuestion: the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?\n\nAnswer: $\\text{Banker's Gain} = \\left( \\text{Present Worth} \\times \\text{Rate} \\times \\text{Time} \\right) - \\left( \\text{True Discount} \\times \\text{Time} \\right)$; $\\text{True Discount} = \\text{Present Worth} \\times \\frac{\\text{Rate} \\times \\text{Time}}{1 + \\text{Rate} \\times \\text{Time}}$."
        system_content_cot = "You are given a question. To answer the question, you think step by step. **Any LaTeX expressions should be wrapped between single dollar signs, e.g., $x^2$**. You should number each step and each step should be only one line. Please use a line break symbol between steps. The final answer to the question should start with \"The answer is ...\", and should be placed at the final step. Be concise and less than 512 tokens.\n\nExample:\n\nQuestion:\naverage age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .\n\nAnswer:\n1. The average age of students at the adult school was initially $A_{\\text{old}} = 40$ years.\n2. There were $N_{\\text{new}} = 120$ new students with an average age of $A_{\\text{new}} = 32$ years.\n3. After the new students joined, the average age decreased by 4 years, making $\\text{New Average Age} = A_{\\text{old}} - 4 = 36$ years.\n4. Let $N_{\\text{old}}$ be the number of original students at the school. Then the total age for the original students is $40N_{\\text{old}}$.\n5. The total age for the new students is $120 \\times 32 = 3840$ years.\n6. The total number of students after the new students joined is $N_{\\text{old} + 120$.\n7. The total age of all students after the new students joined is $40N_{\\text{old}} + 3840$.\n8. The new average age is 36 years. Using the formula for the new average age, we have $36 = \\frac{40N_{\\text{old}} + 3840}{N_{\\text{old}} + 120}$.\n9. Solving the equation $36N_{\\text{old}} + 4320 = 40N_{\\text{old}} + 3840$ leads to $4N_{\\text{old}} = 480$ and hence $N_{\\text{old}} = 120$.\n10. The number of students after the new students joined is $N_{\\text{old}} + N_{\\text{new}} = 120 + 120 = 240$.\n11. The answer is 240."

        # Load the api key
        with open (f"{args.api_config_file_path}", "r") as f:
            config = json.load(f)
            openai_api_key = config["openai_api_key"]
            togetherai_api_key = config["togetherai_api_key"]

        # Load data
        df_all = get_init_df(dataset_name)
        total_samples = len(df_all)

        # Initialize models
        cot_generator_gpt4o = CoTGenerator(
            model_name="gpt-4o",
            chat_history=[
                ("system", system_content_cot)
            ],
            api_key=openai_api_key
        )
        premise_generator_gpt4o = PremiseGenerator(
            model_name="gpt-4",
            chat_history=[
                ("system", system_content_premise)
            ],
            api_key=openai_api_key
        )
        perturbation_generator = PerturbationGenerator(
            model_name="gpt-4o-mini",
            api_key=openai_api_key
        )
        cot_evaluator = CoTEvaluator(api_key=openai_api_key)

        model_names = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "Qwen/Qwen2-72B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "gpt-4o-mini",
        ]
        
        total_usage = {}

        # Open output file in append mode
        with open(f"./pcd_data/mix/test_{4*sample_size}_perturbed.jsonl", "a") as f_out:
            # Iterate over the datafram
            progress_bar = tqdm(range(sample_size - count), desc="Starting...", unit="sample")
            for _ in progress_bar:
                processed = {}
                while df_idx < total_samples:
                    row = df_all.iloc[df_idx]
                    query = "Question: " + row["question"]
                    answer = row["correct_answer"]
                    df_idx += 1
                    # Update the description with the latest values of df_idx and count
                    progress_bar.set_description(f"Index: {df_idx}, Count: {count}")

                    try:
                        # Simulate human knowledge
                        gpt4o_cot_list = cot_generator_gpt4o.generate_cot_list(
                            query,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            number_of_outputs=args.number_of_outputs
                        )

                        if len(gpt4o_cot_list["c_0"]) < 6:
                            continue  # Skip if gpt-4o cannot generate enough steps

                        eval_result = cot_evaluator.evaluate_cot_list(gpt4o_cot_list, answer)
                        if not eval_result["c_0"]["overall_correct"]:
                            continue

                        # Generate premises and perturbations
                        premise_dict = {}
                        perturbed_premise_dict = {}

                        premise_gpt4o = premise_generator_gpt4o.generate_premises(
                            query,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            number_of_outputs=args.number_of_outputs
                        )[0]
                        if "Answer: " in premise_gpt4o:
                            premise_gpt4o = premise_gpt4o.split("Answer: ")[1].strip()

                        perturbed_premise_gpt4o = perturbation_generator.generate_perturbation(
                            premise_gpt4o,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            number_of_outputs=args.number_of_outputs
                        )
                        perturbation_generator.clear_history()
                        if not perturbed_premise_gpt4o:
                            continue

                        premise_dict["human"] = premise_gpt4o
                        perturbed_premise_dict["human"] = perturbed_premise_gpt4o

                        continue_flag = False
                        # Probe LLM's knowledge
                        for model_name in model_names:
                            api_key = togetherai_api_key if "gpt" not in model_name else openai_api_key
                            premise_generator_test_llm = PremiseGenerator(
                                model_name=model_name,
                                chat_history=[
                                    ("system", system_content_premise)
                                ],
                                api_key=api_key
                            )
                            premise = premise_generator_test_llm.generate_premises(
                                query,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                number_of_outputs=args.number_of_outputs
                            )[0]
                            for model, usage in premise_generator_test_llm.get_usage().items():
                                if model not in total_usage:
                                    total_usage[model] = {
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "total_tokens": 0
                                    }
                                total_usage[model]["prompt_tokens"] += usage["prompt_tokens"]
                                total_usage[model]["completion_tokens"] += usage["completion_tokens"]
                                total_usage[model]["total_tokens"] += usage["total_tokens"]
                            if "Answer: " in premise:
                                premise = premise.split("Answer: ")[1].strip()

                            perturbed_premise = perturbation_generator.generate_perturbation(
                                premise,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                number_of_outputs=args.number_of_outputs
                            )
                            perturbation_generator.clear_history()
                            if not perturbed_premise:
                                continue_flag = True
                                break

                            if continue_flag:
                                break

                            premise_dict[model_name] = premise
                            perturbed_premise_dict[model_name] = perturbed_premise

                        if continue_flag:
                            continue

                        processed = {
                            "question": row["question"],
                            "correct_answer": row["correct_answer"],
                            "premise": premise_dict,
                            "perturbed_premise": perturbed_premise_dict,
                            "perturbation_model": "gpt-4o-mini",
                        }
                        # Save the processed data
                        f_out.write(json.dumps(processed) + "\n")
                        f_out.flush()  # Ensure data is written to disk
                        count += 1
                        # Update the description with the latest values of df_idx and count
                        progress_bar.set_description(f"Index: {df_idx}, Count: {count}")
                        break

                    except Exception as e:
                        print(f"An error occurred at index {df_idx - 1}: {e}")
                        continue  # Continue with the next iteration

                if df_idx >= total_samples:
                    print("Reached the end of the dataset.")
                    break
                
        for model, usage in premise_generator_gpt4o.get_usage().items():
            if model not in total_usage:
                total_usage[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            total_usage[model]["prompt_tokens"] += usage["prompt_tokens"]
            total_usage[model]["completion_tokens"] += usage["completion_tokens"]
            total_usage[model]["total_tokens"] += usage["total_tokens"]
            
        for model, usage in cot_generator_gpt4o.get_usage().items():
            if model not in total_usage:
                total_usage[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            total_usage[model]["prompt_tokens"] += usage["prompt_tokens"]
            total_usage[model]["completion_tokens"] += usage["completion_tokens"]
            total_usage[model]["total_tokens"] += usage["total_tokens"]
            
        for model, usage in perturbation_generator.get_usage().items():
            if model not in total_usage:
                total_usage[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            total_usage[model]["prompt_tokens"] += usage["prompt_tokens"]
            total_usage[model]["completion_tokens"] += usage["completion_tokens"]
            total_usage[model]["total_tokens"] += usage["total_tokens"]
            
        for model, usage in total_usage.items():
            print(f"Model: {model}")
            print(f"Prompt tokens: {usage['prompt_tokens']}")
            print(f"Completion tokens: {usage['completion_tokens']}")
            print(f"Total tokens: {usage['total_tokens']}")
            print()

def parse_args():
    parser = argparse.ArgumentParser(description="Data Generation Script")
    parser.add_argument("--dataset_names", nargs='+', default=["gsm8k", "math", "mathqa", "metamath"], help="Dataset name")
    parser.add_argument('--api_config_file_path', type=str, default="./api_key/config.json", help='API config file path')
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--start_count", type=int, default=0, help="Starting count for resuming")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index in the dataframe")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top k for sampling")
    parser.add_argument("--number_of_outputs", type=int, default=1, help="Number of outputs to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    generate_examples(args)

if __name__ == "__main__":
    main()
