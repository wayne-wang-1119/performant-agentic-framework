import numpy as np
from scipy import stats
import os
import pandas as pd

NAIVE_INPUT_FILE = os.path.join(os.path.dirname(__file__), "data/eval_naive.csv")
BASE_INPUT_FILE = os.path.join(os.path.dirname(__file__), "data/eval_base.csv")
PAF_INPUT_FILE = os.path.join(os.path.dirname(__file__), "data/eval_paf.csv")

# Load and preprocess dataset

naive_df = pd.read_csv(NAIVE_INPUT_FILE)
base_df = pd.read_csv(BASE_INPUT_FILE)
paf_df = pd.read_csv(PAF_INPUT_FILE)

naive_scores = np.array(naive_df["naive_semantic_similarity"])  # shape (50,)
base_scores = np.array(base_df["base_semantic_similarity"])  # shape (50,)
paf_scores = np.array(paf_df["optimized_semantic_similarity"])  # shape (50,)

alpha = 0.05  # significance level

# Paired t-test for PAF vs Naive
t_stat_naive, p_val_naive_two_sided = stats.ttest_rel(paf_scores, naive_scores)
# Adjust for one-sided test (alternative: PAF > Naive)
p_val_naive_one_sided = (
    p_val_naive_two_sided / 2 if t_stat_naive > 0 else 1 - p_val_naive_two_sided / 2
)

print(
    f"PAF vs Naive: t-statistic = {t_stat_naive}, one-sided p-value = {p_val_naive_one_sided}"
)
if p_val_naive_one_sided < alpha:
    print(f"Result: Statistically significant (p < {alpha}). PAF is better than Naive.")
else:
    print(f"Result: Not statistically significant (p ≥ {alpha}).")

# Paired t-test for PAF vs Base
t_stat_base, p_val_base_two_sided = stats.ttest_rel(paf_scores, base_scores)
# Adjust for one-sided test (alternative: PAF > Base)
p_val_base_one_sided = (
    p_val_base_two_sided / 2 if t_stat_base > 0 else 1 - p_val_base_two_sided / 2
)

print(
    f"PAF vs Base: t-statistic = {t_stat_base}, one-sided p-value = {p_val_base_one_sided}"
)
if p_val_base_one_sided < alpha:
    print(f"Result: Statistically significant (p < {alpha}). PAF is better than Base.")
else:
    print(f"Result: Not statistically significant (p ≥ {alpha}).")
