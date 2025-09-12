import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiment_log.csv")

print("Alle Ergebnisse:")
print(df)

# Sortiere nach bester Val-PPL
best = df.sort_values("best_val_ppl").head(5)
print("\nTop 5 Modelle (nach niedrigster Val-PPL):")
print(best)

# Plot Val-PPL gegen Embedding/Hidden Size
plt.figure(figsize=(8,5))
for model in df["model"].unique():
    subset = df[df["model"] == model]
    plt.scatter(subset["hidden_dim"], subset["best_val_ppl"], label=model)

plt.xlabel("Hidden Dim")
plt.ylabel("Best Val PPL")
plt.title("Vergleich der Modelle")
plt.legend()
plt.savefig("model_comparison.png")
plt.close()

# Plot Generalization Gap
    plt.figure(figsize=(8,5))
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        plt.scatter(subset["hidden_dim"], subset["gen_gap"], label=model)

    plt.xlabel("Hidden Dim")
    plt.ylabel("Generalization Gap (Val - Train Loss)")
    plt.title("Overfitting-Vergleich")
    plt.legend()
    plt.savefig("generalization_gap.png")
    plt.close()

    print("Analyse abgeschlossen -> Ergebnisse in 'model_comparison.png' und 'generalization_gap.png'")
