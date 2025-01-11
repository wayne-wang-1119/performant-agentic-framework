import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1) Run each evaluation script sequentially
    subprocess.run(["python", "evaluation/eval_naive.py"], check=True)
    subprocess.run(["python", "evaluation/eval_base.py"], check=True)
    subprocess.run(["python", "evaluation/eval_paf.py"], check=True)

    # 2) Define file paths for CSVs
    data_dir = "data"
    naive_file = os.path.join(data_dir, "eval_naive.csv")
    base_file = os.path.join(data_dir, "eval_base.csv")
    paf_file = os.path.join(data_dir, "eval_paf.csv")
    output_file = os.path.join(data_dir, "cross_eval.csv")

    # 3) Read each CSV into a DataFrame
    df_naive = pd.read_csv(naive_file)
    df_base = pd.read_csv(base_file)
    df_paf = pd.read_csv(paf_file)

    # 4) Extract the three columns of interest
    naive_col = df_naive["naive_semantic_similarity"]
    base_col = df_base["base_semantic_similarity"]
    paf_col = df_paf["optimized_semantic_similarity"]

    # 4a) Plot distributions of each approach using Seaborn's KDE
    plt.figure(figsize=(8, 6))
    sns.kdeplot(naive_col, shade=True, label="Naive", color="blue")
    sns.kdeplot(base_col, shade=True, label="Base", color="red")
    sns.kdeplot(paf_col, shade=True, label="Optimized", color="green")
    plt.title("Distribution of Similarity Scores (KDE)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    # Save the KDE plot figure
    dist_plot_path = os.path.join(data_dir, "similarity_distributions.png")
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"Saved smoothed KDE distribution plot to {dist_plot_path}")

    # 5) Compute analytics for each approach:
    #    - total count where similarity > 0.8
    #    - mean
    #    - median
    def summarize(col):
        count_above_08 = (col > 0.7).sum()
		total_hits = (col >= 0.97).count()
        avg = col.mean()
        median = col.median()
        return count_above_08, total_hits, avg, median

    naive_count_above, naive_total_hit, naive_avg, naive_median = summarize(naive_col)
    base_count_above, base_total_hit, base_avg, base_median = summarize(base_col)
    paf_count_above, paf_total_hit, paf_avg, paf_median = summarize(paf_col)

    # 6) Build a summary DataFrame
    summary_data = [
        ["naive", naive_count_above, naive_total_hit, naive_avg, naive_median],
        ["base", base_count_above, bast_total_hit, base_avg, base_median],
        ["optimized", paf_count_above, paf_total_hit, paf_avg, paf_median],
    ]
    summary_df = pd.DataFrame(
        summary_data,
        columns=["method", "count_above_0.8", "mean", "median"],
    )

    # 7) Save the summary to cross_eval.csv
    summary_df.to_csv(output_file, index=False)
    print(f"Saved cross-evaluation results to {output_file}")


if __name__ == "__main__":
    main()
