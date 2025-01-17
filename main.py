import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import run

# Constants
DATA_DIR = "data"
NAIVE_FILE = os.path.join(DATA_DIR, "eval_naive.csv")
BASE_FILE = os.path.join(DATA_DIR, "eval_base.csv")
PAF_FILE = os.path.join(DATA_DIR, "eval_paf.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "cross_eval.csv")
DIST_PLOT_PATH = os.path.join(DATA_DIR, "similarity_distributions.png")


def run_evaluation_scripts():
    """Run all evaluation scripts sequentially."""
    scripts = ["eval_naive", "eval_base", "eval_paf"]
    for script in scripts:
        run(["python", "-m", f"evaluation.{script}"], check=True)


def summarize(col):
    """Compute summary statistics for a similarity column."""
    return {
        "count_above_0.8": (col > 0.8).sum(),
        "total_complete_hit": (col >= 0.97).sum(),
        "mean": col.mean(),
        "median": col.median(),
    }


def plot_similarity_distributions(data):
    """Plot KDE distributions for similarity scores."""
    plt.figure(figsize=(8, 6))
    colors = {"naive": "blue", "base": "red", "optimized": "green"}
    for label, col in data.items():
        sns.kdeplot(col, shade=True, label=label.capitalize(), color=colors[label])
    plt.title("Distribution of Similarity Scores (KDE)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(DIST_PLOT_PATH)
    plt.close()
    print(f"Saved smoothed KDE distribution plot to {DIST_PLOT_PATH}")


def main():
    # Step 1: Run evaluation scripts
    run_evaluation_scripts()

    # Step 2: Read CSVs into DataFrames
    df_naive = pd.read_csv(NAIVE_FILE)
    df_base = pd.read_csv(BASE_FILE)
    df_paf = pd.read_csv(PAF_FILE)

    # Step 3: Extract relevant columns for similarity scores
    naive_col = df_naive["naive_semantic_similarity"]
    base_col = df_base["base_semantic_similarity"]
    paf_col = df_paf["optimized_semantic_similarity"]

    # Step 4: Plot distributions
    plot_similarity_distributions({"naive": naive_col, "base": base_col, "optimized": paf_col})

    # Step 5: Compute summaries
    summaries = {
        "naive": summarize(naive_col),
        "base": summarize(base_col),
        "optimized": summarize(paf_col),
    }

    # Step 6: Build and save summary DataFrame
    summary_df = pd.DataFrame(
        [
            ["naive", summaries["naive"]["total_complete_hit"], summaries["naive"]["count_above_0.8"],
             summaries["naive"]["mean"], summaries["naive"]["median"]],
            ["base", summaries["base"]["total_complete_hit"], summaries["base"]["count_above_0.8"],
             summaries["base"]["mean"], summaries["base"]["median"]],
            ["optimized", summaries["optimized"]["total_complete_hit"], summaries["optimized"]["count_above_0.8"],
             summaries["optimized"]["mean"], summaries["optimized"]["median"]],
        ],
        columns=["method", "total_complete_hit", "count_above_0.8", "mean", "median"],
    )
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cross-evaluation results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
