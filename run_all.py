import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
COMPACT_FILE = os.path.join(RESULTS_DIR, "results_compact.csv")

def run_cmd(cmd):
    print(f"\n=== Running: {cmd} ===\n")
    subprocess.run(cmd, shell=True, check=True)

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # RNN/LSTM Modelle (main.py)
    run_cmd("python3 main.py")

    # Transformer von Scratch
    run_cmd("python3 transformer/transformer_main.py")

    # Transformer Fine-Tuning (Pretrained)
    run_cmd("python3 transformer_pretrained/transformer_finetune.py")

    print("\nAll runs finished!")

    # Ergebnisse laden
    if not os.path.isfile(COMPACT_FILE):
        print("Keine Ergebnisse gefunden:", COMPACT_FILE)
        return

    df = pd.read_csv(COMPACT_FILE)
    print("\nErgebnisse geladen:\n", df.tail())

    # plot accurady
    plt.figure(figsize=(8,5))
    df.plot(x="model", y="best_val_acc", kind="bar", legend=False)
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Model Comparison: Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "compare_accuracy.png"))
    plt.close()

    # plot bleu
    plt.figure(figsize=(8,5))
    df.plot(x="model", y="best_val_bleu", kind="bar", legend=False)
    plt.ylabel("BLEU Score")
    plt.title("Model Comparison: BLEU")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "compare_bleu.png"))
    plt.close()

    # plot loss
    plt.figure(figsize=(8,5))
    df.plot(x="model", y="best_val_loss", kind="bar", legend=False)
    plt.ylabel("Validation Loss")
    plt.title("Model Comparison: Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "compare_loss.png"))
    plt.close()

    print("\n✅ Vergleichsplots gespeichert in:", RESULTS_DIR)

if __name__ == "__main__":
    run_all()
