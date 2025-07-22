import pandas as pd 
import numpy as np
import functions as f 
import cv2
import matplotlib.pyplot as plt
import os

validation_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/validation_true_pred.csv')
test_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/test_true_pred.csv')
train_true_pred_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/train_true_pred.csv')

outliers_validation_df = f.find_spatial_outliers(validation_true_pred_df,'x', 'y', 'width', 'height', 'x_pred', 'y_pred', 'w_pred', 'h_pred', iou_col = 'IoU', pid_col='pid', series_col='dcm_series')
outliers_test_df = f.find_spatial_outliers(test_true_pred_df, 'x', 'y', 'width', 'height', 'x_pred', 'y_pred', 'w_pred', 'h_pred', iou_col = 'IoU', pid_col='pid', series_col='dcm_series')
outliers_train_df = f.find_spatial_outliers(train_true_pred_df, 'x', 'y', 'width', 'height', 'x_pred', 'y_pred', 'w_pred', 'h_pred', iou_col = 'IoU', pid_col='pid', series_col='dcm_series')


outliers_validation_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/immagini/" + outliers_validation_df["uid_slice"] + ".png"
outliers_test_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/immagini/" + outliers_test_df["uid_slice"] + ".png"
outliers_train_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/immagini/" + outliers_train_df["uid_slice"] + ".png"


outliers_validation_df.to_csv('/gwpool/users/bscotti/tesi/csv/outliers_validation.csv', index = False)
outliers_test_df.to_csv('/gwpool/users/bscotti/tesi/csv/outliers_test.csv', index = False)
outliers_train_df.to_csv('/gwpool/users/bscotti/tesi/csv/outliers_train.csv', index = False)


print(f"PID: {outliers_validation_df['pid']}, Series: {outliers_validation_df['dcm_series']}, IoU Scores:\n{outliers_validation_df['IoU']}")
common_slices = set(validation_true_pred_df['uid_slice']).intersection(set(train_true_pred_df['uid_slice']))
print(f"Common slices between validation and train: {common_slices}")



output_dir = '/gwpool/users/bscotti/tesi/grafici/train/outliers1'
os.makedirs(output_dir, exist_ok=True)

for index, row in outliers_train_df.iterrows(): 
    image_path = row["image_path"]
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Bounding Box Ground Truth (verde)
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde
    
    # Bounding Box Predetto (rosso)
    xp, yp, wp, hp = int(row["x_pred"]), int(row["y_pred"]), int(row["w_pred"]), int(row["h_pred"])
    cv2.rectangle(image, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)  # Rosso

    # scrivo l'area sul bb true
    area = round(row["area_true"], 2)
    label = f"area: {area}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    

    pid = row["pid"]
    pid_text =f"pid: {pid}"
    cv2.putText(image, pid_text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    filter = row["dcm_series"]
    filter_text =f"filter: {filter}"
    cv2.putText(image, filter_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)


    output_path = os.path.join(output_dir, row['uid_slice']+'.png')
    plt.imsave(output_path, image)
