import functions as f
import pandas as pd 
import numpy as np 
import os 
import cv2

model_path = '/gwpool/users/bscotti/tesi/train/weights/best.pt'
validation_images = '/gwpool/users/bscotti/tesi/dati/dataset/images/validation'
test_images = '/gwpool/users/bscotti/tesi/dati/dataset/images/test'
train_images = '/gwpool/users/bscotti/tesi/dati/dataset/images/train'
filtered_annotations_df = pd.read_csv('/gwpool/users/bscotti/tesi/csv/filtered_annotations.csv') # METTERE FILTERED ANNOTATIONS IN SSH


# faccio l'inferenza su validation e test set utilizzando il valore di conf ottimizzato in precedenza
val_pred_df = f.get_yolo_predictions(model_path, validation_images)
train_pred_df = f.get_yolo_predictions(model_path, train_images)
test_pred_df = f.get_yolo_predictions(model_path, test_images)

train_pred_df.to_csv('/gwpool/users/bscotti/tesi/csv/train_pred.csv', index = False)
val_pred_df.to_csv('/gwpool/users/bscotti/tesi/csv/val_pred.csv', index = False)
test_pred_df.to_csv('/gwpool/users/bscotti/tesi/csv/test_pred.csv', index = False)


# definisco il csv della ground truth 
ground_truth_df = filtered_annotations_df[['pid', 'dcm_series', 'uid_slice', 'x', 'y', 'width', 'height']]

# unisco le predizioni alla ground truth
train_true_pred_df = f.preprocess_and_merge(train_pred_df, ground_truth_df)
validation_true_pred_df = f.preprocess_and_merge(val_pred_df, ground_truth_df)
test_true_pred_df = f.preprocess_and_merge(test_pred_df, ground_truth_df)

train_true_pred_df = f.calculate_iou(train_true_pred_df)
validation_true_pred_df = f.calculate_iou(validation_true_pred_df)
test_true_pred_df= f.calculate_iou(test_true_pred_df)

train_true_pred_df.to_csv('/gwpool/users/bscotti/tesi/csv/train_true_pred.csv', index = False)
test_true_pred_df.to_csv('/gwpool/users/bscotti/tesi/csv/test_true_pred.csv', index = False)
validation_true_pred_df.to_csv('/gwpool/users/bscotti/tesi/csv/validation_true_pred.csv', index = False)

# calcolo della intersection over union  
results_val, validation_iou_df = f.compute_iou_stats(validation_true_pred_df)
results_ts, test_iou_df = f.compute_iou_stats(test_true_pred_df)
# f.plot_and_save_iou(test_iou_df, "/gwpool/users/bscotti/tesi/grafici/test_iou.png")
# f.plot_and_save_iou(validation_iou_df, "/gwpool/users/bscotti/tesi/grafici/val_iou.png")


