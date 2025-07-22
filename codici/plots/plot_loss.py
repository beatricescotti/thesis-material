import pandas as pd
import matplotlib.pyplot as plt

# Leggo il file CSV con i log
log_path = "/gwpool/users/bscotti/tesi/train_poisson/train8/results.csv" 
df = pd.read_csv(log_path)

# Seleziono le metriche di interesse
epochs = df['epoch']
train_box = df['train/box_loss']
val_box = df['val/box_loss']
train_cls = df['train/cls_loss']
val_cls = df['val/cls_loss']
train_dfl = df['train/dfl_loss']
val_dfl = df['val/dfl_loss']

# Imposto la figura con 3 subplot
plt.figure(figsize=(18, 5))

# Box loss
plt.subplot(1, 3, 1)
plt.plot(epochs, train_box, label='Train Box Loss', color='blue', marker='o')
plt.plot(epochs, val_box, label='Val Box Loss', color='cyan', marker='x')
plt.xlabel("Epoche")
plt.ylabel("Box Loss")
plt.title("Box Loss (Train vs Val)")
plt.grid()
plt.legend()

# Class loss
plt.subplot(1, 3, 2)
plt.plot(epochs, train_cls, label='Train Class Loss', color='green', marker='o')
plt.plot(epochs, val_cls, label='Val Class Loss', color='lime', marker='x')
plt.xlabel("Epoche")
plt.ylabel("Class Loss")
plt.title("Class Loss (Train vs Val)")
plt.grid()
plt.legend()

# DFL loss
plt.subplot(1, 3, 3)
plt.plot(epochs, train_dfl, label='Train DFL Loss', color='red', marker='o')
plt.plot(epochs, val_dfl, label='Val DFL Loss', color='orange', marker='x')
plt.xlabel("Epoche")
plt.ylabel("DFL Loss")
plt.title("DFL Loss (Train vs Val)")
plt.grid()
plt.legend()

plt.tight_layout()

# Salvo il grafico
output_path = "/gwpool/users/bscotti/tesi/grafici/losses_plot_8poisson.png"
plt.savefig(output_path)
print(f"Plot salvato in {output_path}")

plt.show()