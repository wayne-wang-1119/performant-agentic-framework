import numpy as np
from scipy import stats
import os
import pandas as pd

# Define file paths
NAIVE_INPUT_FILE = os.path.join(os.path.dirname(__file__), "data/eval_naive.csv")
BASE_INPUT_FILE = os.path.join(os.path.dirname(__file__), "data/eval_base.csv")
PAF_INPUT_FILE = os.path.join(os.path.dirname(__file__), "data/eval_paf.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results.txt")


# Load and preprocess dataset
def load_scores(file_path, column_name):
    """Loads the semantic similarity scores from a CSV file."""
    df = pd.read_csv(file_path)
    return np.array(df[column_name])


naive_scores = load_scores(NAIVE_INPUT_FILE, "naive_semantic_similarity")  # shape (50,)
base_scores = load_scores(BASE_INPUT_FILE, "base_semantic_similarity")  # shape (50,)
paf_scores = load_scores(PAF_INPUT_FILE, "optimized_semantic_similarity")  # shape (50,)


# Paired t-test function
def perform_paired_ttest(scores_a, scores_b, alpha=0.05):
    """
    Perform a one-sided paired t-test to check if scores_a > scores_b.

    Parameters:
    - scores_a: np.array, first set of scores
    - scores_b: np.array, second set of scores
    - alpha: float, significance level

    Returns:
    - t_stat: float, t-statistic
    - p_val_one_sided: float, one-sided p-value
    - result: str, statistical significance result
    """
    t_stat, p_val_two_sided = stats.ttest_rel(scores_a, scores_b, nan_policy="omit")
    p_val_one_sided = p_val_two_sided / 2 if t_stat > 0 else 1 - p_val_two_sided / 2
    result = (
        f"Statistically significant (p < {alpha})"
        if p_val_one_sided < alpha
        else f"Not statistically significant (p ≥ {alpha})"
    )
    return t_stat, p_val_one_sided, result


# Set significance level
alpha = 0.05

# Prepare results for saving
results = []

# 1. Base vs Naive
t_stat, p_val, result = perform_paired_ttest(base_scores, naive_scores, alpha)
base_vs_naive = (
    f"Base vs Naive: t-statistic = {t_stat:.4f}, one-sided p-value = {p_val:.4f}\n"
    f"Result: {result}. Base is {'better' if p_val < alpha else 'not better'} than Naive.\n"
)
results.append(base_vs_naive)
print(base_vs_naive)

# 2. PAF vs Naive
t_stat, p_val, result = perform_paired_ttest(paf_scores, naive_scores, alpha)
paf_vs_naive = (
    f"PAF vs Naive: t-statistic = {t_stat:.4f}, one-sided p-value = {p_val:.4f}\n"
    f"Result: {result}. PAF is {'better' if p_val < alpha else 'not better'} than Naive.\n"
)
results.append(paf_vs_naive)
print(paf_vs_naive)

# 3. PAF vs Base
t_stat, p_val, result = perform_paired_ttest(paf_scores, base_scores, alpha)
paf_vs_base = (
    f"PAF vs Base: t-statistic = {t_stat:.4f}, one-sided p-value = {p_val:.4f}\n"
    f"Result: {result}. PAF is {'better' if p_val < alpha else 'not better'} than Base.\n"
)
results.append(paf_vs_base)
print(paf_vs_base)

# Save results to a text file with UTF-8 encoding
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.writelines(results)

print(f"Results saved to {OUTPUT_FILE}")
