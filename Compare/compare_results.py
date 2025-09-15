# compare_results.py
import pandas as pd
import os

base_dir = "/content/drive/MyDrive/DLAM_Project/results"

# compact
compact_file = os.path.join(base_dir, "results_compact.csv")
if os.path.exists(compact_file):
    df = pd.read_csv(compact_file)
    print("\n=== Accuracy Comparison ===")
    print(df[["model","best_val_acc"]].sort_values("best_val_acc", ascending=False))

    print("\n=== BLEU Comparison ===")
    print(df[["model","best_val_bleu"]].sort_values("best_val_bleu", ascending=False))

    print("\n=== EM Comparison ===")
    print(df[["model","best_val_em"]].sort_values("best_val_em", ascending=False))

    print("\n=== Jaccard Comparison ===")
    print(df[["model","best_val_jacc"]].sort_values("best_val_jacc", ascending=False))

# details
detailed_file = os.path.join(base_dir, "results_detailed.csv")
if os.path.exists(detailed_file):
    df_detail = pd.read_csv(detailed_file)
    print("\n=== Runtime (s) ===")
    print(df_detail[["model","train_time"]].sort_values("train_time"))

    print("\n=== GPU usage (MB) ===")
    print(df_detail[["model","gpu_mem_MB"]].sort_values("gpu_mem_MB"))

# summary
summary_file = os.path.join(base_dir, "results_summary.csv")
if os.path.exists(summary_file):
    df_sum = pd.read_csv(summary_file)
    print("\n=== Full Summary ===")
    print(df_sum.sort_values("val_acc", ascending=False))
