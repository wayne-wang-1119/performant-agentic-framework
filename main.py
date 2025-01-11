import os
import pandas as pd
import subprocess


def main():
    # 1) Run each evaluation script sequentially
    # Make sure your paths are correct relative to this script's location
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

    # 5) Compute analytics for each approach
    #     - total count < 0.7
    #     - mean
    #     - median
    def summarize(col):
        count_below_07 = (col < 0.7).sum()
        avg = col.mean()
        median = col.median()
        return count_below_07, avg, median

    naive_count_below, naive_avg, naive_median = summarize(naive_col)
    base_count_below, base_avg, base_median = summarize(base_col)
    paf_count_below, paf_avg, paf_median = summarize(paf_col)

    # 6) Build a summary DataFrame
    summary_data = [
        ["naive", naive_count_below, naive_avg, naive_median],
        ["base", base_count_below, base_avg, base_median],
        ["optimized", paf_count_below, paf_avg, paf_median],
    ]
    summary_df = pd.DataFrame(
        summary_data,
        columns=["method", "count_below_0.7", "mean", "median"],
    )

    # 7) Save the summary to cross_eval.csv
    summary_df.to_csv(output_file, index=False)
    print(f"Saved cross-evaluation results to {output_file}")


if __name__ == "__main__":
    main()
