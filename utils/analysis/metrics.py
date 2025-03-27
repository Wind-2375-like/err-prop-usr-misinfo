import numpy as np

def perturbation_ratio_given_correct(df, cij="prfx_q", pij="prfx_q", evaluation_type="overall_correct", perturbation_role=None):
    # Same logic as your original function
    temp_df = df.copy()
    count_nominator = 0
    count_denominator = 0
    seed = 42
    seeds = [seed + i for i in range(len(df))]

    for idx, row in temp_df.iterrows():
        ci = np.array([int(v["overall_correct"]) for _, v in row[cij].items()])
        if pij == cij:
            np.random.seed(seeds[idx])
            pi = np.random.permutation(ci)
        else:
            if evaluation_type == "overall_correct":
                if "original" in pij:
                    pi = np.array([int(v["overall_correct"]) for _, v in row[pij].items()])
                else:
                    pi = np.array([int(v["overall_correct"]) for _, v in row[pij][perturbation_role].items()])
            else:
                if "original" in pij:
                    pi = np.array([int(v["evaluation_type"] == evaluation_type) for _, v in row[pij].items()])
                else:
                    pi = np.array([int(v["evaluation_type"] == evaluation_type) for _, v in row[pij][perturbation_role].items()])

        count_nominator += np.sum(ci * pi)
        count_denominator += np.sum(ci)

    return count_nominator/count_denominator if count_denominator != 0 else 0
