import numpy as np
import pandas as pd
import os
import cv2
import matplotlib as plt
from functions import compute_iou

validation_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/validation_true_pred.csv')
test_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/test_true_pred.csv')
train_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/train_true_pred.csv')

# aggiungo path delle immagini 
validation_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/validation/" + validation_true_pred_df["uid_slice"] + ".png"
train_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/train/" + train_true_pred_df["uid_slice"] + ".png"
test_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/test/" + test_true_pred_df["uid_slice"] + ".png"


def compute_average_iou_per_group(df):
    grouped = df.groupby(['pid', 'dcm_series'])  
    updated_df = pd.DataFrame(columns=df.columns)  

    for (pid, series), group in grouped:
        num_boxes = len(group)
        if num_boxes <= 2:
            updated_df = pd.concat([updated_df, group], ignore_index=True)
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

        # Seleziona i bounding box considerati outlier
        outliers = group[group['mean_IoU'] < 0.10]
        non_outliers = group[group['mean_IoU'] >= 0.10]

        if not outliers.empty and not non_outliers.empty:
            # Calcola il bounding box medio dai non-outlier
            mean_box = non_outliers[['x_pred', 'y_pred', 'w_pred', 'h_pred']].mean().values
            
            # Sostituisci gli outlier con il bounding box medio
            group.loc[group['mean_IoU'] < 0.10, ['x_pred', 'y_pred', 'w_pred', 'h_pred']] = mean_box

            print(f"PID: {pid}, Serie: {series} | {len(outliers)} outlier(s) sostituiti con bounding box medio {mean_box}")

        updated_df = pd.concat([updated_df, group.drop(columns=['mean_IoU'])], ignore_index=True)

    return updated_df


test_outliers2_df = compute_average_iou_per_group(test_true_pred_df)
validation_outliers2_df = compute_average_iou_per_group(validation_true_pred_df)
train_outliers2_df = compute_average_iou_per_group(train_true_pred_df)
print(test_outliers2_df)
print(train_outliers2_df)
print(validation_outliers2_df)


import os
import cv2

output_dir = '/gwpool/users/bscotti/tesi/grafici/validation/outliers2/replacement_10'
os.makedirs(output_dir, exist_ok=True)

for index, row in train_outliers2_df.iterrows(): 
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

    # Salva l'immagine con OpenCV
    output_path = os.path.join(output_dir, row['uid_slice'] + '.png')
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # OpenCV salva in BGR