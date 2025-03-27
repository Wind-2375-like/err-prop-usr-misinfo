# Unraveling Error Propagation of User Misinformation in LLM Reasoning

This is the official implementation for error propagation of user misinformation in LLM reasoning.

We develop a pipeline to systematically evaluate the impact of user misinformation on LLM reasoning, analyzing correction behaviors and their effectiveness. 

Our findings indicate that simply instructing models to correct misinformation is insufficient, highlighting the importance of fine-tuning on correction-specific data and early factual corrections. 

Our study provides insights for designing more robust LLMs to better handle misinformed inputs.

## 1. Setup

### 1.1 Environment Setup

We recommend you to create a new [conda](https://docs.conda.io/en/latest/) environment for running codes in the repository. Replace `{your_path}` in `err_prop_usr_misinfo.yml` with your actual conda installation path. Then run the following commands:

```bash
conda env create -f err_prop_usr_misinfo.yml
conda activate err_prop_usr_misinfo
```

### 1.2 API Key Setup

In this project, we need to use [OpenAI API](https://platform.openai.com/docs/overview), [Together AI API](https://docs.together.ai/docs/quickstart), and [Huggingface User Access Token](https://huggingface.co/docs/hub/en/security-tokens) to access different language models. Follow their tutorials to create API keys and put them in `api_key/config.json`. See the GPT-4o-mini-sft API key in 3.2.

```json
{
    "openai_api_key": "...",
    "togetherai_api_key": "...",
    "huggingface_api_key": "...",
    "gpt-4o-mini-sft": "...
}
```

## 2. Data Preprocessing

```bash
python scripts/precompute_bootstrap_results.py
```

We start from raw testing data saved in `raw_data/{dataset_name}/test.jsonl`. The details of preprocessing are in Appendix D. The preprocessing process is implemented in the `get_init_df` function in `scripts/test_data_generation/main.py`.

After preprocessing, we retain only questions where the equation generation model (gpt-4-0613) produces correct answers to ensure the reliability of ground-truth equations. Additionally, to exclude overly simple questions, we filter out those with fewer than 5 CoT steps in their solutions.

Then we simulate user misinformation by generating correct and relevant equations and then perturbing them using common human error patterns. Details are in Section 3.2 and Appendix A.

To generate the processed testing data, run the following command:

```bash
python test_data_generation/main.py --dataset_names gsm8k math mathqa metamath --api_config_file_path api_key/config.json --sample_size 100 --temperature 0.7 --top_p 0.7 --top_k 50 --number_of_outputs 1 
```

(Sample size is set to 100 but the total number of questions is 100*4(datasets)=400.)

The processed data is saved in `pcd_data/mix/test_400_perturbed.jsonl`.

## 3. Experiments

### 3.1 Overview of Experiments

First, we analyze how user misinformation affects the model reasoning accuracy and process with following comparative studies and evaluations:
1. Final Answer Accuracy: we evaluate the correctness of the final answer, using K-Acc, detailed in Section 3.3 and Section 5.1.
2. Correction and Misinformation-Following Behaviors: we implement several automatic verifiers to analyze the reasoning behaviors of different models, detailed in Section 3.3 and Section 5.2.

We observe that LLMs are vulnerable to misinformation in user instructions and struggle to correct it. To help LLMs correct misinformation, we study the effectiveness of two methods:
1. Explicit Correction Instructions: we add relevant prompts when the input has misinformation, detailed in Section 3.3 and Section 6.1.
2. Fine-tuning for Correction: we fine-tune the model on correction-specific data, detailed in Section 3.3 and Section 6.2.

Beyond methods, we further explore how correction itself affects final answer accuracy, we conduct a controlled study by enforcing in:
1. Correction Behaviors: we analyze the effect of corrections and no corrections. For corrections, we further explore the impact of factual and non-factual corrections, detailed in Section 3.3 and Section 7.1.
2. Different Positions of Reasoning Steps: for factual corrections, we explore the impact of positions of reasoning steps where corrections happen, detailed in Section 3.3 and Section 7.2.

### 3.2 Generate Reasoning Steps in Original and Misinformed Settings

Performance is evaluated under two conditions: without misinformation (original) and with misinformation (misinformed) to assess whether strong reasoning models can be misled.

To evaluate model performance corresponding to Section 5 and Section 6.1, run the following command:

```bash
python run_experiments.py --model_name {model_name} --dataset_name mix --sample_size 400 --temperature 0.7 --top_p 0.7 --top_k 50 --number_of_outputs 5 --api_config_file_path api_key/config.json
```

The `{model_name}` could be `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.2-11B-Vision-Instruct`, `Llama-3.2-90B-Vision-Instruct`, `Mixtral-8x7B-Instruct-v0.1`, `Mixtral-8x22B-Instruct-v0.1`, `Qwen2-72B-Instruct`, `gpt-4o-mini`. The prediction includes `prfx_q` (original performance, without misinformation), `prfx_q_prfx_pert` (misinformed performance, with misinformation), and `prfx_q_prfx_pert_both` (misinformed performance with explicit correction instruction) and results are saved in `pcd_data/mix/test_400_perturbed_premise_{model_name}.jsonl`.

To evaluate model performance corresponding to Section 6.2, first, run the first part (Fine-tuning) of cells in `demonstration/finetune.ipynb`. You will finetune a GPT-4o-mini model and get the model id. Save the id in `api_key/config.json`. Then run the following command:

```bash
python run_experiments_finetune.py --model_name gpt-4o-mini-sft --dataset_name mix --sample_size 400 --temperature 0.7 --top_p 0.7 --top_k 50 --number_of_outputs 5 --api_config_file_path api_key/config.json
```

The prediction includes `base_original` (original performance), `base_misinformed` (misinformed performance), `inst_original` (original performance with explicit instructions), `inst_misinformed` (misinformed performance with explicit instruction), `ft_original` (original performance with fine-tuning), `ft_misinformed` (misinformed performance with fine-tuning), `inst_ft_original` (original performance with both explicit instructions and fine-tuning), and `inst_ft_misinformed` (misinformed performance with both explicit instructions and fine-tuning). Results are saved in `pcd_data/mix/test_4oo_perturbed_premise_{model_name}.jsonl`. Note that the correction-specific data is saved in `pcd_data/finetune/correction_training_set.jsonl`. We will release the data collection code in the future.

To evaluate model performance corresponding to Section 7, run the following command:

```bash
python run_experiments_intervention.py --model_name {model_name} --dataset_name mix --sample_size 400 --temperature 0.7 --top_p 0.7 --top_k 50 --number_of_outputs 5 --api_config_file_path api_key/config.json
```

The `{model_name}` could be `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.2-11B-Vision-Instruct`. The prediction includes `prfx_q`, `prfx_q_prfx_pert`, and `prfx_q_prfx_pert_both`, the same as `run_experiments.py`. There are also `prfx_q_prfx_pert_pos` (misinformed performance with factual corrections at different positions), `prfx_q_prfx_pert_corr_bad` (misinformed performance with non-factual corrections), `prfx_q_prfx_pert_point_out_only_bad` (misinformed performance with no corrections). Results are saved in `pcd_data/mix/test_400_perturbed_premise_{model_name}_correction.jsonl`.

### 3.3 Evaluate Reasoning Steps with Verifiers

We implement several automatic verifiers to analyze the reasoning behaviors of different models. To evaluate the correction and misinformation-following behaviors corresponding to Section 5.2, run the following command:

```bash
python scripts/run_evaluation.py --model_names "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo" "Qwen/Qwen2-72B-Instruct" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x22B-Instruct-v0.1" "gpt-4o-mini" "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-11B-Vision-Instruct" --output_path "./exp_results/eval/test_400_perturbed_premise_evaluated.pkl"
```

To evaluate the correction behaviors in Section 6.1, run the following command:

```bash
python scripts/run_evaluation.py --model_names "meta-llama/Llama-3.2-1B-Instruct" --output_path "./exp_results/eval/test_400_perturbed_premise_evaluated_1b.pkl"
```

To evaluate the correction behaviors in Section 6.2, follow the Evaluation part in `demonstration/finetune.ipynb`.

We also have three annotators to manually evaluate the correction behaviors to validate the effectiveness of automatic verifiers. Follow `demonstration/annotation.ipynb` and we gather the results in `exp_results/eval/test_400_perturbed_premise_evaluated_annotated_final.pkl`.

## 4. Results and Analysis

To reproduce the results in the paper, first run the following command:

```bash
python scripts/precompute_bootstrap_results.py
```

Results are in `exp_results/`.

Then follow `demonstration/result_visualization.ipynb` to plot figures, which are saved in `figures/`.

For the fine-tuning results, follow the Plot Sankey Graph part in `demonstration/finetune.ipynb`.

## 5. Citation

If you find this repository helpful, please cite our paper:

```
TODO
```
