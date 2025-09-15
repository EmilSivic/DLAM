import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from IPython.display import Image, display

RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
COMPACT_FILE = os.path.join(RESULTS_DIR, "results_compact.csv")

def run_cmd(cmd):
    print(f"\n=== Running: {cmd} ===\n")
    subprocess.run(cmd, shell=True, check=True)

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # runs
    run_cmd("python3 main.py")                             # LSTM/RNN
    run_cmd("python3 transformer/transformer_main.py")     # Transformer Scratch
    run_cmd("python3 transformer_pretrained/transformer_finetune.py")  # Fine-Tuning

    print("\n=== All runs finished! ===")

    # Ergebnisse laden
    if not os.path.isfile(COMPACT_FILE):
        print("⚠️ Keine Ergebnisse gefunden:", COMPACT_FILE)
        return

    df = pd.read_csv(COMPACT_FILE)
    print("\nErgebnisse geladen:\n", df.tail())

    # accuracy
    plt.figure(figsize=(8,5))
    df.plot(x="model", y="best_val_acc", kind="bar", legend=False)
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Model Comparison: Accuracy")
    plt.tight_layout()
    acc_path = os.path.join(RESULTS_DIR, "compare_accuracy.png")
    plt.savefig(acc_path)
    plt.close()
    print("✅ Accuracy-Plot gespeichert:", acc_path)

    # BLEU
    plt.figure(figsize=(8,5))
    df.plot(x="model", y="best_val_bleu", kind="bar", legend=False)
    plt.ylabel("BLEU Score")
    plt.title("Model Comparison: BLEU")
    plt.tight_layout()
    bleu_path = os.path.join(RESULTS_DIR, "compare_bleu.png")
    plt.savefig(bleu_path)
    plt.close()
    print("✅ BLEU-Plot gespeichert:", bleu_path)

    # loss
    plt.figure(figsize=(8,5))
    df.plot(x="model", y="best_val_loss", kind="bar", legend=False)
    plt.ylabel("Validation Loss")
    plt.title("Model Comparison: Loss")
    plt.tight_layout()
    loss_path = os.path.join(RESULTS_DIR, "compare_loss.png")
    plt.savefig(loss_path)
    plt.close()
    print("✅ Loss-Plot gespeichert:", loss_path)

    # Lokale Artefakte (Losskurven + Confusion Matrices)
    print("\n=== Zusätzliche Artefakte ===")
    for f in glob.glob("*.png"):
        print("Gefunden:", f)
        try:
            display(Image(filename=f))
        except Exception:
            pass

    for f in glob.glob(os.path.join(RESULTS_DIR, "*_confusion_matrix.png")):
        print("Gefunden:", f)
        try:
            display(Image(filename=f))
        except Exception:
            pass

    print("\nVergleichsplots & Artefakte gespeichert in:", RESULTS_DIR)

if __name__ == "__main__":
    run_all()
