import os
import cv2
import numpy as np
import pandas as pd
from functions import compute_iou
import optuna
import matplotlib.pyplot as plt

validation_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/validation_true_pred.csv')
test_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/test_true_pred.csv')
train_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/train_true_pred.csv')

# aggiungo path delle immagini 
validation_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dati_medici/dataset/images/validation/" + validation_true_pred_df["uid_slice"] + ".png"
train_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dati_medici/dataset/images/train/" + train_true_pred_df["uid_slice"] + ".png"
test_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dati_medici/dataset/images/test/" + test_true_pred_df["uid_slice"] + ".png"


# Funzione per calcolare gli outlier e restituire il numero totale di outlier
def compute_average_iou_per_group(df, iou_threshold):
    grouped = df.groupby(['pid', 'dcm_series'])
    outlier_count = 0  # Contatore per il numero di outlier

    for (pid, series), group in grouped:
        num_boxes = len(group)
        if num_boxes <= 2:
            continue  

        iou_means = []
        for i in range(num_boxes):
            iou_values = []
            box1 = group.iloc[i][['x_pred', 'y_pred', 'w_pred', 'h_pred']].values
            
            for j in range(num_boxes):
                if i != j:
                    box2 = group.iloc[j][['x_pred', 'y_pred', 'w_pred', 'h_pred']].values
                    iou = compute_iou(box1, box2)
                    iou_values.append(iou)

            mean_iou = np.mean(iou_values)
            iou_means.append(mean_iou)

        group = group.copy()
        group['mean_IoU'] = iou_means

        # Conta il numero di outlier per questa soglia
        outlier_count += (group['mean_IoU'] < iou_threshold).sum()

    return outlier_count  # Obiettivo: minimizzare il numero di outlier


# Funzione obiettivo per Optuna
def objective(trial):
    iou_threshold = trial.suggest_float("iou_threshold", 0.05, 0.5)  # Esploriamo soglie tra 0.05 e 0.5
    outlier_count = compute_average_iou_per_group(validation_true_pred_df, iou_threshold)
    # Registriamo anche i dati per il grafico
    trials_data.append((iou_threshold, outlier_count))
    return outlier_count  # Minimizzare il numero di outlier

# Lista per memorizzare i dati da visualizzare
trials_data = []

# Creiamo lo studio Optuna per trovare la soglia ottimale
study = optuna.create_study(direction="minimize")  # Minimizziamo il numero di outlier
study.optimize(objective, n_trials=50)

# Miglior soglia trovata
best_threshold = study.best_params["iou_threshold"]
print(f"Miglior soglia IoU trovata: {best_threshold:.2f}")

# Creiamo il grafico
iou_thresholds, outlier_counts = zip(*trials_data)

# Creazione della cartella se non esiste
output_dir = '/gwpool/users/bscotti/tesi/grafici'
os.makedirs(output_dir, exist_ok=True)

# Salviamo il grafico nella cartella
plt.figure(figsize=(10, 6))
plt.plot(iou_thresholds, outlier_counts, marker='o', linestyle='-', color='b')
plt.title("Numero di Outlier vs Soglia IoU durante l'ottimizzazione")
plt.xlabel("Soglia IoU")
plt.ylabel("Numero di Outlier")
plt.grid(True)

# Salvataggio del grafico
output_path = os.path.join(output_dir, 'iou_vs_outliers_optimization.png')
plt.savefig(output_path)
plt.close()


"""
def compute_average_iou_per_group(df):
    grouped = df.groupby(['pid', 'dcm_series']) 
    outliers_list = []  # Lista per salvare i nuovi outliers con bounding box medi

    for (pid, series), group in grouped:
        num_boxes = len(group)
        if num_boxes <= 2:
            continue  

        iou_means = []

        for i in range(num_boxes):
            iou_values = []
            box1 = group.iloc[i][['x_pred', 'y_pred', 'w_pred', 'h_pred']].values
            
            for j in range(num_boxes):
                if i != j:
                    box2 = group.iloc[j][['x_pred', 'y_pred', 'w_pred', 'h_pred']].values
                    iou = compute_iou(box1, box2)
                    iou_values.append(iou)
            
            mean_iou = np.mean(iou_values)
            iou_means.append(mean_iou)

        group = group.copy()
        group['mean_IoU'] = iou_means

        # Calcoliamo il bounding box medio dei box validi
        valid_boxes = group[group['mean_IoU'] >= 0.10][['x_pred', 'y_pred', 'w_pred', 'h_pred']]
        if not valid_boxes.empty:
            mean_box = valid_boxes.mean().values
        else:
            continue  # Nessun bounding box valido → non possiamo calcolare il box medio

        # Selezioniamo gli outliers e aggiungiamo i bounding box medi
        outliers = group[group['mean_IoU'] < 0.10].copy()  # Copy per sicurezza
        if not outliers.empty:
            outliers[['x_fixed', 'y_fixed', 'w_fixed', 'h_fixed']] = mean_box
            outliers_list.append(outliers)

    # Creiamo il DataFrame finale con tutti gli outliers aggiornati
    if outliers_list:
        return pd.concat(outliers_list, ignore_index=True)
    else:
        return pd.DataFrame(columns=df.columns)  # Se nessun outlier, restituiamo un DF vuoto

# Calcoliamo gli outliers e i bounding box medi
#test_outliers2_df = compute_average_iou_per_group(test_true_pred_df)
validation_outliers2_df = compute_average_iou_per_group(validation_true_pred_df)
#train_outliers2_df = compute_average_iou_per_group(train_true_pred_df)

# Output directory
output_dir = '/gwpool/users/bscotti/tesi/grafici/validation/outliers2/replacement'
os.makedirs(output_dir, exist_ok=True)

# Plottiamo solo gli outliers con il bounding box medio
for index, row in validation_outliers2_df.iterrows(): 
    image_path = row["image_path"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Bounding Box Ground Truth (verde)
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "T", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Bounding Box Outlier (rosso)
    xp, yp, wp, hp = int(row["x_pred"]), int(row["y_pred"]), int(row["w_pred"]), int(row["h_pred"])
    cv2.rectangle(image, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)
    cv2.putText(image, "P (outlier)", (xp, yp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Bounding Box Corretto (blu)
    xf, yf, wf, hf = int(row["x_fixed"]), int(row["y_fixed"]), int(row["w_fixed"]), int(row["h_fixed"])
    cv2.rectangle(image, (xf, yf), (xf + wf, yf + hf), (0, 0, 255), 2)
    cv2.putText(image, "F (fixed)", (xf, yf - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Aggiungiamo PID e serie
    pid = row["pid"]
    cv2.putText(image, f"pid: {pid}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    dcm_series = row["dcm_series"]
    cv2.putText(image, f"series: {dcm_series}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    # Salviamo l'immagine
    output_path = os.path.join(output_dir, row['uid_slice'] + '.png')
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    """