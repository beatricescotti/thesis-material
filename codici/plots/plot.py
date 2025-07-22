import pandas as pd
import matplotlib.pyplot as plt

# Leggo il file CSV con i log
log_path = "/gwpool/users/bscotti/tesi/train_6jet/train/results.csv" 
df = pd.read_csv(log_path)

# Seleziono le metriche di interesse
epochs = df['epoch']
precision = df['metrics/precision(B)']
recall = df['metrics/recall(B)']
map50 = df['metrics/mAP50(B)']
map5095 = df['metrics/mAP50-95(B)']

# Imposto la figura con 4 subplot
plt.figure(figsize=(20, 5))

# Precision (blu)
plt.subplot(1, 4, 1)
plt.plot(epochs, precision, label='Precision (B)', color='blue', marker='o')
plt.xlabel("Epoche")
plt.ylabel("Precision")
plt.title("Precision (B)")
plt.grid()

# Recall (verde)
plt.subplot(1, 4, 2)
plt.plot(epochs, recall, label='Recall (B)', color='green', marker='o')
plt.xlabel("Epoche")
plt.ylabel("Recall")
plt.title("Recall (B)")
plt.grid()

# mAP@0.5 (giallo)
plt.subplot(1, 4, 3)
plt.plot(epochs, map50, label='mAP@0.5 (B)', color='gold', marker='o')
plt.xlabel("Epoche")
plt.ylabel("mAP@0.5")
plt.title("mAP@0.5 (B)")
plt.grid()

# mAP@0.5:0.95 (rosso)
plt.subplot(1, 4, 4)
plt.plot(epochs, map5095, label='mAP@0.5:0.95 (B)', color='red', marker='o')
plt.xlabel("Epoche")
plt.ylabel("mAP@0.5:0.95")
plt.title("mAP@0.5:0.95 (B)")
plt.grid()

plt.tight_layout()

# Salvo il grafico
output_path = "/gwpool/users/bscotti/tesi/grafici/metrics_plot_jet6.png"
plt.savefig(output_path)
print(f"Plot salvato in {output_path}")

plt.show()