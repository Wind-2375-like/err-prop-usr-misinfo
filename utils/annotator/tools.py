import numpy as np
from collections import Counter
from utils.data_loader import load_precomputed_results

def flatten_list_of_lists(list_of_lists):
    ret = []
    for l in list_of_lists:
        ret.extend(l)
    return ret

def load_evaluated_results_with_sample_size(pickle_path, sample_size):
    # Load a pickle file with precomputed bootstrap results
    data = load_precomputed_results(pickle_path)
    
    def wrap_with_dict(bool_col):
        return {"label": bool_col}

    data["overall_correct"] = data["overall_correct"].apply(wrap_with_dict)
    data["point_out_correct"] = data["point_out_correct"].apply(wrap_with_dict)
        
    def filter_out_truncated_answer(row):
        return 'answer' in row['output'][-1] and 'answer' in row['point_out_output'][-1]
    
    data = data[data.apply(filter_out_truncated_answer, axis=1)].sample(sample_size, random_state=42).reset_index(drop=True)
    return data

def evaluate_results(df_sample, annotator_name):
    # Compute F1
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    annotators = ["label", annotator_name]
    answer_verifier = [[], []]
    detection_verifier = [[], []]
    correction_verifier = [[], []]
    perturbation_verifier = [[], []]
    position_verifier = [[], []]

    for i, annotator in enumerate(annotators):
        answer_verifier[i] = df_sample["overall_correct"].apply(lambda x: x[annotator]).values.tolist() + df_sample["point_out_correct"].apply(lambda x: x[annotator]).values.tolist()
        detection_verifier[i] = df_sample["detection"].apply(lambda x: x[annotator]).values.tolist() + df_sample["detection_prompting"].apply(lambda x: x[annotator]).values.tolist()
        correction_verifier[i] = df_sample["correction"].apply(lambda x: x[annotator]).values.tolist() + df_sample["correction_prompting"].apply(lambda x: x[annotator]).values.tolist()
        perturbation_verifier[i] = df_sample["perturbation"].apply(lambda x: x[annotator]).values.tolist() + df_sample["perturbation_prompting"].apply(lambda x: x[annotator]).values.tolist()
        position_verifier[i] = flatten_list_of_lists(df_sample["detection_positions"].apply(lambda x: x[annotator]).values.tolist()) + flatten_list_of_lists(df_sample["detection_prompting_positions"].apply(lambda x: x[annotator]).values.tolist())
    
    return answer_verifier[1], detection_verifier[1], correction_verifier[1], perturbation_verifier[1], position_verifier[1], f1_score(answer_verifier[0], answer_verifier[1], average="weighted"), f1_score(detection_verifier[0], detection_verifier[1], average="weighted"), f1_score(correction_verifier[0], correction_verifier[1], average="weighted"), f1_score(perturbation_verifier[0], perturbation_verifier[1], average="weighted"), f1_score(position_verifier[0], position_verifier[1], average="weighted")

def compute_fleiss_kappa(annotations_dict):
    # Step 1: Create the frequency matrix
    # Convert annotations to a list of lists (rows: items, columns: annotators)
    annotations_list = list(annotations_dict.values())
    num_items = len(annotations_list[0])  # Number of examples
    num_annotators = len(annotations_dict)  # Number of annotators
    
    # Transpose annotations to group by items
    annotations_per_item = np.array(annotations_list).T.tolist()

    # Unique categories (labels)
    categories = set(val for annotations in annotations_list for val in annotations)
    num_categories = len(categories)

    # Build frequency matrix
    frequency_matrix = np.zeros((num_items, num_categories), dtype=int)
    category_index = {cat: idx for idx, cat in enumerate(sorted(categories))}

    for item_idx, item_annotations in enumerate(annotations_per_item):
        item_counts = Counter(item_annotations)
        for category, count in item_counts.items():
            frequency_matrix[item_idx][category_index[category]] = count

    # Step 2: Compute P_j (proportion for each category)
    category_totals = frequency_matrix.sum(axis=0)
    total_annotations = frequency_matrix.sum()
    P_j = category_totals / total_annotations

    # Step 3: Compute P_i (proportion of agreement per item)
    P_i = []
    for row in frequency_matrix:
        n_i = row.sum()
        if n_i > 1:  # Avoid division by zero
            P_i.append(((row ** 2).sum() - n_i) / (n_i * (n_i - 1)))
        else:
            P_i.append(0)
    P_i = np.array(P_i)

    # Step 4: Compute average agreement (P-bar) and expected agreement (P_e)
    P_bar = P_i.mean()
    P_e = (P_j ** 2).sum()

    # Step 5: Compute Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e) if 1 - P_e != 0 else 0
    return kappa
