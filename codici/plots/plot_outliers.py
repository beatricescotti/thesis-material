import numpy as np
import matplotlib.pyplot as plt
import os

# Dati della tabella
iou_thresholds = np.array([1, 2, 3])

train_outliers = np.array([90, 325, 762])
val_outliers   = np.array([8, 32, 63])
test_outliers  = np.array([10, 38, 72])

# Normalizzazione rispetto al massimo valore
train_norm = (1 - train_outliers / 4797)*100
val_norm   = (1 - val_outliers / 469)*100
test_norm  =(1 - test_outliers / 487)*100

# Creazione del plot
plt.figure(figsize=(8, 6))
plt.plot(iou_thresholds, train_norm, '-o', label="Train", linewidth=2)
plt.plot(iou_thresholds, val_norm, '-s', label="Validation", linewidth=2)
plt.plot(iou_thresholds, test_norm, '-d', label="Test", linewidth=2)

# Personalizzazione
plt.xlabel("Sigma Threshold")
plt.ylabel("Totale - Outliers in % ")
plt.legend()
plt.grid(True)

# Creazione della cartella se non esiste
output_folder = "/gwpool/users/bscotti/tesi/grafici"
os.makedirs(output_folder, exist_ok=True)

# Salvataggio in PNG e PDF
plt.savefig(os.path.join(output_folder, "outliers_plot_distance.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_folder, "outliers_plot_distance.pdf"), bbox_inches="tight")

plt.close()  # Chiude la figura per non visualizzarla direttamente