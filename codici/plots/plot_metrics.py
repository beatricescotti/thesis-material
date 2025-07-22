import pandas as pd
import matplotlib.pyplot as plt

# Leggi il file CSV con i log
log_path = "/gwpool/users/bscotti/tesi/train_poisson/train8/results.csv"
df = pd.read_csv(log_path)

# Seleziona le metriche di interesse
epochs = df['epoch']
precision = df['metrics/precision(B)']
recall = df['metrics/recall(B)']
map50 = df['metrics/mAP50(B)']
map50_95 = df['metrics/mAP50-95(B)']

# Imposta la figura
plt.figure(figsize=(10, 6))

# Plot delle metriche
plt.plot(epochs, precision, label='Precision', color='red', marker='o')
plt.plot(epochs, recall, label='Recall', color='yellow', marker='x')
plt.plot(epochs, map50, label='mAP@50', color='blue', marker='s')
plt.plot(epochs, map50_95, label='mAP@50-95', color='green', marker='d')

# Personalizzazioni del grafico
plt.xlabel("Epoche")
plt.ylabel("Valore")
plt.title("Metriche di Valutazione")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Salvo il grafico
output_path = "/gwpool/users/bscotti/tesi/grafici/metrics_plot_8poisson.png"
plt.savefig(output_path)
print(f"Plot salvato in {output_path}")

plt.show()