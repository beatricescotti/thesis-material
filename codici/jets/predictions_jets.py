import pandas as pd 
import numpy as np 
import os 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from ultralytics import YOLO
import glob

model_path = '/gwpool/users/bscotti/tesi/training_jet/train10/weights/best.pt'
validation_images = '/gwpool/users/bscotti/tesi/dati/dataset/images/validation'
test_images = '/gwpool/users/bscotti/tesi/dati/dataset/images/test'
train_images = '/gwpool/users/bscotti/tesi/dati/dataset/images/train'
image_path = '/gwpool/users/bscotti/tesi/dati/folder_1_ttbar_qcd_hbb_100_jet'
csv_path = '/gwpool/users/bscotti/tesi/csv/bounding_boxes_hbb_100_jet.csv'

annotations_df = pd.read_csv(csv_path)

def get_yolo_predictions(model_path, dataset_path, conf=0.3): 
    model = YOLO(model_path)
    image_paths = glob.glob(os.path.join(dataset_path, '*'))

    predictions = []
    # Itero su tutte le immagini
    for image_path in image_paths:
        image_name = os.path.basename(image_path)  # Estrai il nome dell'immagine una sola volta
        results = model(image_path, conf=conf)  # Esegui l'inferenza

        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # Estrai le coordinate del box
                confidence_level = box.conf[0]

                predictions.append({
                    'image_path': image_path, 
                    'image_name': image_name,
                    'x_pred': x_min,  # Coordinate predette
                    'y_pred': y_min,
                    'w_pred': x_max - x_min,
                    'h_pred': y_max - y_min,
                    'conf' : confidence_level
                })

    # Crea il DataFrame con tutte le predizioni
    pred_df = pd.DataFrame(predictions)

    return pred_df


validation_pred = get_yolo_predictions(model_path, validation_images)
test_pred = get_yolo_predictions(model_path, test_images)
train_pred = get_yolo_predictions(model_path, train_images)


def preprocess_and_merge(df1, df2, column='image_name'):

    df1 = df1.copy()
    df2 = df2.copy()

    df1[column] = df1[column].str.split('/').str[-1].str.replace('.jpg', '', regex=False)
    df2[column] = df2[column].str.split('/').str[-1].str.replace('.jpg', '', regex=False)

    merged_df = pd.merge(df1, df2, on=column, how='inner')

    return merged_df

val_true_pred_df = preprocess_and_merge(annotations_df, validation_pred)
train_true_pred_df = preprocess_and_merge(annotations_df, train_pred)
test_true_pred_df = preprocess_and_merge(annotations_df, test_pred)



output_dir = '/gwpool/users/bscotti/tesi/grafici/jet/100/validation'
os.makedirs(output_dir, exist_ok=True)

for index, row in val_true_pred_df.iterrows(): 
    image_path = row["image_path"]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Errore nel caricare l'immagine: {image_path}")
        continue  # Salta questa immagine se non può essere caricata
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Bounding Box Ground Truth (verde)
    x, y, w, h = int(row["_x_min"]), int(row["bbox_y_min"]), int(row["bbox_width"]), int(row["bbox_height"])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "hbb", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Bounding Box Outlier (rosso)
    xp, yp, wp, hp = int(row["x_pred"]), int(row["y_pred"]), int(row["w_pred"]), int(row["h_pred"])
    cv2.rectangle(image, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)
    cv2.putText(image, "hbb1", (xp, yp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Aggiungiamo PID e serie
    pid = row["image_name"]
    cv2.putText(image, f"image: {pid}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Usa direttamente image_name come nome del file
    output_path = os.path.join(output_dir, row['image_name'])  # Il nome del file include già l'estensione

    # Salva l'immagine
    if image.shape[0] > 0 and image.shape[1] > 0:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        print(f"Immagine vuota: {image_path}")