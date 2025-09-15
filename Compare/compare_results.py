import pandas as pd

base_dir = "/content/drive/MyDrive/DLAM_Project/results"
df = pd.read_csv(f"{base_dir}/results_compact.csv")

print("\n=== Accuracy Comparison ===")
print(df[["model","best_val_acc"]].sort_values("best_val_acc", ascending=False))

print("\n=== BLEU Comparison ===")
print(df[["model","best_val_bleu"]].sort_values("best_val_bleu", ascending=False))

print("\n=== Runtime (s) ===")
df_detail = pd.read_csv(f"{base_dir}/results_detailed.csv")
print(df_detail[["model","train_time","gpu_mem_MB"]].sort_values("train_time"))
