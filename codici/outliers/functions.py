import os
import pandas as pd
import numpy as np
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt 


def get_yolo_predictions(model_path, dataset_path, conf= 0.1): 
    """
    Genera predizioni con un modello YOLO su tutte le immagini di un dataset.
    args:
        model_path: Percorso al file .pt del modello YOLO.
        dataset_path: Percorso alla cartella contenente le immagini su cui fare predizioni.
        conf (float): Confidence threshold per le predizioni. Valore ottimizzato con Optuna.
    
    returns:
        pd.DataFrame: DataFrame con colonne `uid_slice`, `x_pred`, `y_pred`, `w_pred`, `h_pred`.
    """

    model = YOLO(model_path)
    image_paths = glob.glob(os.path.join(dataset_path, '*'))

    predictions = []
    # itero su tutte le immagini
    for image_path in image_paths:
        results = model(image_path, conf=conf)  
        image_name = os.path.basename(image_path) 
        for result in results:
            for box in result.boxes:
                x_centro, y_centro, width, height = box.xywh[0].tolist()  

                predictions.append({
                    'uid_slice': image_path,  
                    'x_pred': (x_centro - width / 2),  # Convertito dal formato YOLO
                    'y_pred': (y_centro - height / 2),
                    'w_pred': width,
                    'h_pred': height,
                })

    pred_df = pd.DataFrame(predictions)

    return pred_df



def preprocess_and_merge(df1, df2, column='uid_slice'):

    df1 = df1.copy()
    df2 = df2.copy()

    df1[column] = df1[column].str.split('/').str[-1].str.replace('.png', '', regex=False)
    df2[column] = df2[column].str.split('/').str[-1].str.replace('.png', '', regex=False)

    merged_df = pd.merge(df1, df2, on=column, how='inner')

    return merged_df



def calculate_iou(df, x='x', y='y', w='width', h='height', 
                  x_pred='x_pred', y_pred='y_pred', w_pred='w_pred', h_pred='h_pred'):
    """
    Calcola l'Intersection over Union (IoU) tra le bounding box reali e quelle predette.
    
    Args:
        df (pd.DataFrame): DataFrame contenente le coordinate e dimensioni delle bounding box.
        x, y, w, h (str): Nomi delle colonne per bounding box reali.
        x_pred, y_pred, w_pred, h_pred (str): Nomi delle colonne per bounding box predette.
    
    Returns:
        pd.DataFrame: DataFrame con colonne aggiunte per 'area_true' e 'IoU'.
    """
    area_true = df[w] * df[h]
    area_pred = df[w_pred] * df[h_pred]

    x1 = np.maximum(df[x], df[x_pred])
    x2 = np.minimum(df[x] + df[w], df[x_pred] + df[w_pred])
    y1 = np.maximum(df[y], df[y_pred])
    y2 = np.minimum(df[y] + df[h], df[y_pred] + df[h_pred])

    area_intersection = np.maximum(0, (x2 - x1)) * np.maximum(0, (y2 - y1))
    area_union = area_true + area_pred - area_intersection

    iou = np.where(area_union > 0, area_intersection / area_union, 0)

    df['area_true'] = area_true
    df['IoU'] = iou

    return df


def compute_iou_stats(df, pid_col='pid', area_col='area_true', iou_col='IoU', top_n=23):
    results = {}
    x_values = []
    y_values = []


    pid_counts = df.groupby(pid_col).size()
    print("\nNumero di rilevamenti per ogni PID:")
    print(pid_counts)

    for k in range(1, top_n + 1, 2):  # Solo valori dispari: 1, 3, 5, ..., 11 
        # seleziono le righe con area maggiore
        area_topk_df = df.groupby(pid_col).apply(lambda x: x.nlargest(k, area_col)).reset_index(drop=True)
        
        # calcolo media iou 
        iou_mean = area_topk_df[iou_col].mean()

        results[f"top{k}"] = {"df": area_topk_df, "iou_mean": iou_mean}

        # aggiungo i valori al df finale
        x_values.append(k)
        y_values.append(iou_mean)


    # creazione df finale
    iou_df = pd.DataFrame({'numero slices considerate': x_values, 'IoU media': y_values})

    return results, iou_df



def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    inter_x1, inter_y1 = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x2, inter_y2 = min(x1_max, x2_max), min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1, area2 = (x1_max - x1_min) * (y1_max - y1_min), (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def enlarge_bbox(x, y, w, h, scale=1.2):
    """Ingrandisce il bounding box del 20%"""
    new_w, new_h = w * scale, h * scale
    new_x, new_y = x - (new_w - w) / 2, y - (new_h - h) / 2  
    return new_x, new_y, new_w, new_h



def find_spatial_outliers(df, x_col, y_col, w_col, h_col, xp_col, yp_col, wp_col, hp_col, iou_col, pid_col='pid', series_col='dcm_series'):
    """
    Trova il bounding box più distante in ogni gruppo (`pid`, `dcm_series`).
    
    1. Trova il bbox con IoU massimo
    2. Lo ingrandisce del 20%
    3. Ricalcola IoU tra il bbox ingrandito e il bounding box predetto
    4. Trova l'outlier con IoU più basso rispetto al bbox ingrandito
    """
    outliers = []
    for (pid, series), group in df.groupby([pid_col, series_col]):
        if len(group) > 1:
            # bounding box con IoU massimo nel gruppo
            best_bbox = group.loc[group[iou_col].idxmax(), [xp_col, yp_col, wp_col, hp_col]].values
            best_bbox_enlarged = enlarge_bbox(*best_bbox)  # ingrandisco del 20%

            #  IoU tra il bbox ingrandito e tutti gli altri predetti nel gruppo
            iou_scores = group.apply(lambda row: compute_iou(best_bbox_enlarged, 
                                                              (row[xp_col], row[yp_col], row[wp_col], row[hp_col])), axis=1)

            # seleziono i bounding box con IoU pari a 0 rispetto al bbox ingrandito
            zero_iou_boxes = group[iou_scores == 0]
            outliers.extend(zero_iou_boxes.to_dict(orient="records"))

    return pd.DataFrame(outliers)




import torch
import torch.nn as nn

class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()

        # primo blocco convoluzionale
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.3)

        # secondo blocco convoluzionale
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        # terzo blocco convoluzionale
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.4)

        # quarto blocco convoluzionale
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(0.4)

        # strati Fully Connected
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 256)
        self.drop_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.drop_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 32)
        self.drop_fc3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(32, 4)  # output finale con (x, y, w, h)

    def forward(self, x):

        x = self.pool1(nn.ReLU()(self.bn2(self.conv2(nn.ReLU()(self.bn1(self.conv1(x)))))))
        x = self.drop1(x)

        x = self.pool2(nn.ReLU()(self.bn4(self.conv4(nn.ReLU()(self.bn3(self.conv3(x)))))))
        x = self.drop2(x)

        x = self.pool3(nn.ReLU()(self.bn6(self.conv6(nn.ReLU()(self.bn5(self.conv5(x)))))))
        x = self.drop3(x)

        x = self.pool4(nn.ReLU()(self.bn8(self.conv8(nn.ReLU()(self.bn7(self.conv7(x)))))))
        x = self.drop4(x)

        x = self.global_pool(x)
        x = self.flatten(x)

        x = nn.ReLU()(self.fc1(x))
        x = self.drop_fc1(x)

        x = nn.ReLU()(self.fc2(x))
        x = self.drop_fc2(x)

        x = nn.ReLU()(self.fc3(x))
        x = self.drop_fc3(x)

        x = self.fc4(x)  # output finale (x, y, w, h)
        return x