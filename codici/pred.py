import pandas as pd 
import numpy as np 
import os 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from ultralytics import YOLO
import glob
import re


model_path = '/gwpool/users/bscotti/tesi/train_6jet/train5/weights/best.pt'
validation_images = '/gwpool/users/bscotti/tesi/dati/dataset_6/images/validation'
test_images = '/gwpool/users/bscotti/tesi/dati/dataset_6/images/test'
train_images = '/gwpool/users/bscotti/tesi/dati/dataset_6/images/train'
#image_path = '/gwpool/users/bscotti/tesi/dati/dati_jet/immagini_50'
csv_path = '/gwpool/users/bscotti/tesi/csv/bounding_boxes_6_classe_1.csv'
csv_path0 = '/gwpool/users/bscotti/tesi/csv/bounding_boxes_6_classe_0.csv'

annotations0_df = pd.read_csv(csv_path0)
annotations1_df = pd.read_csv(csv_path)


def get_yolo_predictions_aggregated(model_path, dataset_path, conf=0.35): 
    model = YOLO(model_path)
    model.to('cuda:1')
    image_paths = glob.glob(os.path.join(dataset_path, '*'))

    class_names = {0: "qcd", 1: "hbb"}  

    rows = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        results = model(image_path, conf=conf)

        row = {'image_path': image_path, 'image_name': image_name}
        i = 0

        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                confidence_level = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names.get(class_id, "unknown")

                # Aggiungi le predizioni alla riga corrente, con suffisso numerico
                row[f'x_pred{i}'] = x_min
                row[f'y_pred{i}'] = y_min
                row[f'w_pred{i}'] = x_max - x_min
                row[f'h_pred{i}'] = y_max - y_min
                row[f'conf{i}'] = confidence_level
                row[f'class_id{i}'] = class_id
                row[f'class_name{i}'] = class_name

                i += 1

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    return pred_df


#validation_pred = get_yolo_predictions_aggregated(model_path, validation_images)
test_pred = get_yolo_predictions_aggregated(model_path, test_images)
#train_pred = get_yolo_predictions_aggregated(model_path, train_images)


model = YOLO(model_path)
model.to('cuda:1')
print("Working directory:", os.getcwd())
metrics = model.val(data='/gwpool/users/bscotti/tesi/dati/dataset_12/dataset.yaml', split='test', conf=0.35)



# Imposta la cartella di output
output_dir = "/gwpool/users/bscotti/tesi/grafici/jet/test_pred6"
os.makedirs(output_dir, exist_ok=True)

# Numero massimo di predizioni (ad esempio da 1 a 6)
max_preds = 7

for idx, row in test_pred.iterrows():
    image_path = row['image_path']  # o il nome corretto della colonna
    if not os.path.exists(image_path):
        continue  # skip se l'immagine non esiste

    image = cv2.imread(image_path)
    if image is None:
        continue  # skip se non si riesce a leggere

    for i in range(1, max_preds + 1):
        x = row.get(f'x_pred{i}')
        y = row.get(f'y_pred{i}')
        w = row.get(f'w_pred{i}')
        h = row.get(f'h_pred{i}')
        class_id = row.get(f'class_id{i}')

        if pd.notna(x) and pd.notna(y) and pd.notna(w) and pd.notna(h):
            # Converte in interi
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'Class {int(class_id) if pd.notna(class_id) else "?"}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    # Salva l'immagine
    image_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, image_name), image)




# === CONFIGURAZIONE ===
gt_df = annotations1_df.copy()
pred_img_dir = "/gwpool/users/bscotti/tesi/grafici/jet/test_pred"
output_dir = "/gwpool/users/bscotti/tesi/grafici/jet/test_pred_with_gt"
os.makedirs(output_dir, exist_ok=True)

# === AGGIUNGI PATH COMPLETO DELLE IMMAGINI ===
gt_df["image_path"] = gt_df["image_name"].apply(lambda name: os.path.join(pred_img_dir, name))
existing_paths = set(glob(os.path.join(pred_img_dir, "*.jpg")))
gt_df = gt_df[gt_df["image_path"].isin(existing_paths)]

# === TROVA TUTTI I GRUPPI DI BOX (qcd1, qcd2, ..., hbb, ...) ===
box_groups = []
for col in gt_df.columns:
    match = re.match(r"(.*)_xmin", col)
    if match:
        box_groups.append(match.group(1))  # es: qcd1, hbb, ecc.

box_groups = sorted(set(box_groups))

# === DISEGNA BOX PER OGNI IMMAGINE ===
for idx, row in gt_df.iterrows():
    image_path = row["image_path"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Immagine non trovata: {image_path}")
        continue

    for group in box_groups:
        x = row.get(f"{group}_xmin")
        y = row.get(f"{group}_ymin")
        w = row.get(f"{group}_width")
        h = row.get(f"{group}_height")

        if pd.notna(x) and pd.notna(y) and pd.notna(w) and pd.notna(h):
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Box GT verde sottile
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Label personalizzata: ad esempio 0 per qcd*, 1 per hbb
            if group.startswith("qcd"):
                label = "0 (qcd)"
            elif group.startswith("hbb"):
                label = "1 (hbb)"
            else:
                label = "?"

            cv2.putText(image, label, (x + w + 10, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Salva immagine annotata
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, image)




# === CONFIGURAZIONE ===
gt_df = annotations0_df.copy()
pred_img_dir = "/gwpool/users/bscotti/tesi/grafici/jet/test_pred"
output_dir = "/gwpool/users/bscotti/tesi/grafici/jet/test_pred_with_gt"
os.makedirs(output_dir, exist_ok=True)

# === AGGIUNGI COLONNA CON PATH IMMAGINE ===
gt_df["image_path"] = gt_df["image_name"].apply(lambda name: os.path.join(pred_img_dir, name))

# === FILTRA SOLO IMMAGINI ESISTENTI ===
existing_paths = set(glob(os.path.join(pred_img_dir, "*.jpg")))
gt_df = gt_df[gt_df["image_path"].isin(existing_paths)]

# === ESTRAI NOMI DEI GRUPPI DI BOX (es. qcd1, qcd2, ecc.) ===
gt_groups = sorted(set(
    re.match(r"(qcd\d+)_xmin", col).group(1)
    for col in gt_df.columns
    if re.match(r"qcd\d+_xmin", col)
))

# === DISEGNA GROUND TRUTH SU IMMAGINI ===
for idx, row in gt_df.iterrows():
    image_path = row["image_path"]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Immagine non trovata: {image_path}")
        continue

    for group in gt_groups:
        x = row.get(f"{group}_xmin")
        y = row.get(f"{group}_ymin")
        w = row.get(f"{group}_width")
        h = row.get(f"{group}_height")
        class_id = row.get(f"{group}_class_id")
        class_name = row.get(f"{group}_class_name")

        if pd.notna(x) and pd.notna(y) and pd.notna(w) and pd.notna(h):
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Box ground truth in verde
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.putText(image, "0 (qcd)", (x + w + 10, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Salva immagine aggiornata
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, image)
