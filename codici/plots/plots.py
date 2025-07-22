import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Carica i DataFrame
validation_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/validation_true_pred.csv')
test_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/test_true_pred.csv')
train_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/train_true_pred.csv')

# Aggiungi path delle immagini
validation_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/validation/" + validation_true_pred_df["uid_slice"] + ".png"
train_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/train/" + train_true_pred_df["uid_slice"] + ".png"
test_true_pred_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dataset/images/test/" + test_true_pred_df["uid_slice"] + ".png"

# Crea una cartella di destinazione se non esiste
output_dir = '/gwpool/users/bscotti/tesi/grafici/val/images_with_bboxes2'
os.makedirs(output_dir, exist_ok=True)

for index, row in validation_true_pred_df.iterrows(): 
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

