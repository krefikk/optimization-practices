import pandas as pd
import matplotlib.pyplot as plt
import glob

# Algoritma adları
algorithm_names = ["Gradient Descent", "Stochastic Gradient Descent", "Adam"]

# Tüm dosyaları yükleme
files = glob.glob("algorithm_*.csv")

# Başlangıç ağırlıkları kümelerini belirleme
initial_weights = set([file.split("_")[3].split(".csv")[0] for file in files])

# Her başlangıç ağırlığı için grafik çizimi
for w_init in initial_weights:
    # Şu başlangıç ağırlığı için grafik çiziminde kullanılacak figür
    plt.figure(figsize=(12, 8))
    
    # Epoch'a göre doğruluk ve kayıp
    plt.subplot(2, 2, 1)
    plt.title(f"Accuracy vs Epoch (w={w_init})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(2, 2, 2)
    plt.title(f"Loss vs Epoch (w={w_init})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Zaman'a göre doğruluk ve kayıp
    plt.subplot(2, 2, 3)
    plt.title(f"Accuracy vs Time (w={w_init})")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Accuracy")

    plt.subplot(2, 2, 4)
    plt.title(f"Loss vs Time (w={w_init})")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Loss")

    # Şu başlangıç ağırlığına ait dosyaları işle
    for file in files:
        # Dosya adından algoritma ve başlangıç ağırlığı bilgilerini al
        parts = file.split("_")
        algorithm_index = int(parts[1])
        algorithm_name = algorithm_names[algorithm_index]
        current_w_init = parts[3].split(".csv")[0]

        # Sadece şu başlangıç ağırlığına uygun dosyaları işle
        if current_w_init != w_init:
            continue

        # Verileri oku
        data = pd.read_csv(file)

        # Doğruluk (Accuracy) - Epoch
        plt.subplot(2, 2, 1)
        plt.plot(data["Epoch"], data["TrainAccuracy"], label=f"Train - {algorithm_name}")
        plt.plot(data["Epoch"], data["TestAccuracy"], label=f"Test - {algorithm_name}", linestyle="--")

        # Kayıp (Loss) - Epoch
        plt.subplot(2, 2, 2)
        plt.plot(data["Epoch"], data["TrainLoss"], label=f"Loss - {algorithm_name}")

        # Doğruluk (Accuracy) - Zaman
        plt.subplot(2, 2, 3)
        plt.plot(data["Time"].cumsum(), data["TrainAccuracy"], label=f"Train - {algorithm_name}")
        plt.plot(data["Time"].cumsum(), data["TestAccuracy"], label=f"Test - {algorithm_name}", linestyle="--")

        # Kayıp (Loss) - Zaman
        plt.subplot(2, 2, 4)
        plt.plot(data["Time"].cumsum(), data["TrainLoss"], label=f"Loss - {algorithm_name}")

    # Her alt grafik için legend ve düzenleme
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.legend(fontsize='small', loc="upper right")

    plt.tight_layout()
    plt.savefig(f"Comparison_w_{w_init}.png")
    plt.show()