import pandas as pd
import matplotlib.pyplot as plt
import os


csv_paths = {
    'senza pre train' : '/gwpool/users/bscotti/tesi/train_dati_andre/padding/train5/results.csv',
    'con il pre train': '/gwpool/users/bscotti/tesi/dati/final_jet/training_curriculum_medical/DLCS/results.csv',
    }

 
# Colonne da plottare
metrics = {
    'train/cls_loss': 'Classification Loss',
    'train/box_loss': 'Box Loss',
    'train/dfl_loss': 'DFL Loss'
}

# Legge tutti i dataframe
dataframes = {}
for name, path in csv_paths.items():
    df = pd.read_csv(path)
    dataframes[name] = df

# Cartella per i plot
output_dir = '/gwpool/users/bscotti/tesi/dati/final_jet/confronto/medical'
os.makedirs(output_dir, exist_ok=True)


# Plot delle metriche per epoca (usando la lunghezza effettiva dei dati)
for col, title in metrics.items():
    plt.figure(figsize=(10, 6))
    for name, df in dataframes.items():
        if col in df.columns:
            y = df[col].dropna()
            x = range(1, len(y) + 1)
            plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel('Epoche')
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fname = col.split('/')[-1].replace('(', '').replace(')', '').replace('-', '_') + '.png'
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()