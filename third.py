import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import glob

# Tüm dosyaları yükleme
files = glob.glob("algorithm_*.csv")

# Her algoritma ve başlangıç ağırlığı için w1:t değerlerini saklamak
all_weights = []

for file in files:
    # Dosya adından algoritma ve başlangıç ağırlığı bilgilerini çıkar
    parts = file.split("_")
    algorithm_index = int(parts[1])

    # w_init değerini "w" harfini çıkararak doğru sayıya dönüştürme
    w_init_str = parts[3].split(".csv")[0]  # w0.00001 gibi bir değer
    w_init = float(w_init_str[1:])  # "w" harfini atlayarak sayıya dönüştür

    # Verileri oku
    data = pd.read_csv(file)

    # Tüm ağırlık vektörlerini al (Epoch başına bir ağırlık vektörü)
    weights = data.iloc[:, 5:].values  # İlk 5 sütun Epoch, TrainLoss, TestAccuracy, TrainAccuracy, Time
    all_weights.append({"algorithm": algorithm_index, "w_init": w_init, "weights": weights})

# Kontrol
print(f"Toplam {len(all_weights)} veri kümesi yüklendi.")

# Algoritma isimleri
algorithm_names = ["Gradient Descent", "Stochastic Gradient Descent", "Adam"]

# Algoritma başına t-SNE uygulama ve görselleştirme
for alg_index in range(3):  # 0: GD, 1: SGD, 2: Adam
    plt.figure(figsize=(12, 8))
    tsne_results = []

    for item in all_weights:
        if item["algorithm"] == alg_index:
            weights = item["weights"]
            tsne = TSNE(n_components=2, random_state=42, perplexity=25)
            tsne_2d = tsne.fit_transform(weights)
            tsne_results.append({
                "w_init": item["w_init"],
                "tsne_2d": tsne_2d
            })
    
    # Algoritmaya ait her başlangıç ağırlığı için t-SNE çıktılarını çizme
    for idx, result in enumerate(tsne_results):
        x = result["tsne_2d"][:, 0]
        y = result["tsne_2d"][:, 1]
        w_init = result["w_init"]
        
        # Başlangıç ağırlığına göre çizim
        plt.plot(x, y, marker="o", label=f"w_init={w_init}")

    # Başlık ve etiketler
    plt.title(f"t-SNE Visualization of w1:t Trajectories ({algorithm_names[alg_index]})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    # Grafik kaydetme ve gösterme
    plt.savefig(f"tsne_trajectories_{algorithm_names[alg_index]}.png")
    plt.show()
