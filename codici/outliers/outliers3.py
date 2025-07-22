
""" METODO 1: distanza tra i centri
1. calcolo il centro di ogni bounding box predetto, x_c  = [x_min + (x_min + width)]/2 , y_c = [y_min+(y_min + height)]/2
2. calcolo la distanza tra i centri per ogni bb predetto e faccio la media 
3. identifico come outlier il box che ha una distanza dal gruppo molto più alta (es 3 sigma)
"""

import pandas as pd 
import numpy as np 
import functions as f 
import cv2
import matplotlib.pyplot as plt
import os

validation_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/validation_true_pred.csv')
test_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/test_true_pred.csv')
train_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/train_true_pred.csv')

# aggiungo path delle immagini 
validation_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/validation/" + validation_true_pred_df["uid_slice"] + ".png"
train_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/train/" + train_true_pred_df["uid_slice"] + ".png"
test_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/test/" + test_true_pred_df["uid_slice"] + ".png"


def compute_outliers_based_on_distance(df):
    grouped = df.groupby(['pid', 'dcm_series']) 
    outliers_df = pd.DataFrame(columns=df.columns)  

    for (pid, series), group in grouped:
        num_boxes = len(group)
        if num_boxes <= 2:
            continue  

        distances = []
        print(f"\n--- PID: {pid}, Serie: {series} ---")

        centroid_list = []
        for i in range(num_boxes):
            box = group.iloc[i][['x_pred', 'y_pred', 'w_pred', 'h_pred']].values
            x_c = box[0] + box[2] / 2
            y_c = box[1] + box[3] / 2
            centroid_list.append((x_c, y_c))

        for i in range(num_boxes):
            dist_values = []
            x1, y1 = centroid_list[i]

            for j in range(num_boxes):
                if i != j:
                    x2, y2 = centroid_list[j]
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    dist_values.append(dist)
                    print(f"  Distanza tra box {i} e {j}: {dist:.3f}")

            mean_dist = np.mean(dist_values)
            distances.append(mean_dist)
            print(f"PID: {pid}, Serie: {series} | Box {i}: Distanza media = {mean_dist:.6f}")

        # Aggiungere le distanze al DataFrame
        group = group.copy()
        group['mean_distance'] = distances

        # Calcolo soglia 3 sigma
        mu = np.mean(distances)
        sigma = np.std(distances)
        threshold = mu + 1.96 * sigma

        # Filtrare gli outlier
        num_outliers = len(group[group['mean_distance'] > threshold])
        print(f"PID: {pid}, Serie: {series} | Aggiunti {num_outliers} outlier con distanza > {threshold:.3f}")

        outliers_df = pd.concat([outliers_df, group[group['mean_distance'] > threshold]], ignore_index=True)

    return outliers_df

test_outliers_dist_df = compute_outliers_based_on_distance(test_true_pred_df)
validation_outliers_dist_df = compute_outliers_based_on_distance(validation_true_pred_df)
train_outliers_dist_df = compute_outliers_based_on_distance(train_true_pred_df)

print("Test:", test_outliers_dist_df)
print("Train:", train_outliers_dist_df)
print("Validation:", validation_outliers_dist_df)
"""
output_dir = '/gwpool/users/bscotti/tesi/grafici/train/outliers1/3sigma'
os.makedirs(output_dir, exist_ok=True)

for index, row in train_outliers_dist_df.iterrows(): 
    image_path = row["image_path"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Bounding Box Ground Truth (verde)
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "T", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Bounding Box Predetto (rosso)
    xp, yp, wp, hp = int(row["x_pred"]), int(row["y_pred"]), int(row["w_pred"]), int(row["h_pred"])
    cv2.rectangle(image, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)
    cv2.putText(image, "P", (xp, yp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # Aggiungiamo PID e filtro
    pid = row["pid"]
    cv2.putText(image, f"pid: {pid}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    filter = row["dcm_series"]
    cv2.putText(image, f"filter: {filter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    output_path = os.path.join(output_dir, row['uid_slice'] + '.png')
    plt.imsave(output_path, image)
    """