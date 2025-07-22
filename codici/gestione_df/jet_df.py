import shutil
import pandas as pd 
import glob
import numpy as np
import random
import os
import shutil
from sklearn.model_selection import train_test_split
import cv2 
import matplotlib.pyplot as plt 


csv_path1= '/gwpool/users/bscotti/tesi/csv/bounding_boxes_6_classe_1.csv'
csv_path0= '/gwpool/users/bscotti/tesi/csv/bounding_boxes_6_classe_0.csv'

annotations1_df = pd.read_csv(csv_path1)
annotations0_df = pd.read_csv(csv_path0)

annotations0_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/bea_imgs_jets/jet_6_classe_0/" + annotations0_df["image_name"]
annotations1_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/bea_imgs_jets/jet_6_classe_1/" + annotations1_df["image_name"]

image_dir = "/gwpool/users/bscotti/tesi/dati/dataset_6/images" 
labels_path = "/gwpool/users/bscotti/tesi/dati/dataset_6/labels" 



# CREAZIONE FILE CONFIGURAZIONE YOLO
yaml_path = "/gwpool/users/bscotti/tesi/dati/dataset_6/dataset.yaml"

# contenuto del file di configurazione
yaml_content = """\
path : /gwpool/users/bscotti/tesi/dati/dataset_6

train: images/train
val: images/validation
test: images/test 

nc: 2
names: ["qcd", "hbb"] 
"""

with open(yaml_path, "w") as file:
    file.write(yaml_content)



# dataset in formato yolo (x_centro, y_centro, width, height) con valori normalizzati tra 0 e 1 e non in formato (x, y, width, height)
# CLASSE 0
img_size = (800, 800, 1)

yolo_annotations_0_df = pd.DataFrame({
    'image_name' : annotations0_df['image_name'],
    'image_path': annotations0_df['image_path'],

    'class_id1': 0,
    'x_center1': (annotations0_df['qcd1_xmin']+ annotations0_df['qcd1_width']/2) / img_size[0],
    'y_center1': (annotations0_df['qcd1_ymin'] + annotations0_df['qcd1_height']/2) / img_size[1],
    'width1': annotations0_df['qcd1_width'] / img_size[0],
    'height1': annotations0_df['qcd1_height'] / img_size[1],

    'class_id2' : 0,
    'x_center2': (annotations0_df['qcd2_xmin']+ annotations0_df['qcd2_width']/2) / img_size[0],
    'y_center2': (annotations0_df['qcd2_ymin'] + annotations0_df['qcd2_height']/2) / img_size[1],
    'width2': annotations0_df['qcd2_width'] / img_size[0],
    'height2': annotations0_df['qcd2_height'] / img_size[1],

    'class_id3': 0,
    'x_center3': (annotations0_df['qcd3_xmin']+ annotations0_df['qcd3_width']/2) / img_size[0],
    'y_center3': (annotations0_df['qcd3_ymin'] + annotations0_df['qcd3_height']/2) / img_size[1],
    'width3': annotations0_df['qcd3_width'] / img_size[0],
    'height3': annotations0_df['qcd3_height'] / img_size[1]
})




# CLASSE 1 
yolo_annotations_1_df = pd.DataFrame({
    'image_name' : annotations1_df['image_name'],
    'image_path': annotations1_df['image_path'],

    'class_id1': 1,
    'x_center1': (annotations1_df['hbb_xmin']+ annotations1_df['hbb_width']/2) / img_size[0],
    'y_center1': (annotations1_df['hbb_ymin'] + annotations1_df['hbb_height']/2) / img_size[1],
    'width1': annotations1_df['hbb_width'] / img_size[0],
    'height1': annotations1_df['hbb_height'] / img_size[1],

    'class_id2' : 0,
    'x_center2': (annotations1_df['qcd2_xmin']+ annotations1_df['qcd2_width']/2) / img_size[0],
    'y_center2': (annotations1_df['qcd2_ymin'] + annotations1_df['qcd2_height']/2) / img_size[1],
    'width2': annotations1_df['qcd2_width'] / img_size[0],
    'height2': annotations1_df['qcd2_height'] / img_size[1],

    'class_id3': 0,
    'x_center3': (annotations1_df['qcd1_xmin']+ annotations1_df['qcd1_width']/2) / img_size[0],
    'y_center3': (annotations1_df['qcd1_ymin'] + annotations1_df['qcd1_height']/2) / img_size[1],
    'width3': annotations1_df['qcd1_width'] / img_size[0],
    'height3': annotations1_df['qcd1_height'] / img_size[1]
})


# CREAZIONE DEI DATASET DI TRAINING VALIDATION E TEST 
train0_df, test_valid0_df = train_test_split(yolo_annotations_0_df, train_size=0.8, random_state=43)
test0_df, validation0_df = train_test_split(test_valid0_df, test_size=0.5, random_state=43)

train1_df, test_valid1_df = train_test_split(yolo_annotations_1_df, train_size=0.8, random_state=43)
test1_df, validation1_df = train_test_split(test_valid1_df, test_size=0.5, random_state=43)



# IMAGES
def move_images(df, split_name):
    # creo la cartella di destinazione se non esiste
    destination_path = os.path.join(image_dir, split_name)
    os.makedirs(destination_path, exist_ok=True)

    for _, row in df.iterrows():
        img_path = row["image_path"]  # percorso dell'immagine
        img_name = row["image_name"]
        new_path = os.path.join(destination_path, img_name)  # nuovo percorso
        shutil.copy(img_path, new_path)  # copia l'immagine nella cartella di destinazione

# Applicazione della funzione per train, validation e test
move_images(train1_df, "train")
move_images(validation1_df, "validation")
move_images(test1_df, "test")

move_images(train0_df, "train")
move_images(validation0_df, "validation")
move_images(test0_df, "test")



# LABELS
def save_annotations(df, subfolder, labels_path="/gwpool/users/bscotti/tesi/dati/dataset_6/labels"):
    output_dir = os.path.join(labels_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        filename = os.path.basename(row['image_name']).replace('.jpg', '.txt')
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            for i in range(1, 4):  # Supponendo che ci siano sempre 3 oggetti per immagine
                class_id = row[f'class_id{i}']
                x_center = row[f'x_center{i}']
                y_center = row[f'y_center{i}']
                width = row[f'width{i}']
                height = row[f'height{i}']

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Annotazioni salvate in {output_dir}")

save_annotations(train0_df, 'train')
save_annotations(validation0_df, 'validation')
save_annotations(test0_df, 'test')

save_annotations(train1_df, 'train')
save_annotations(validation1_df, 'validation')
save_annotations(test1_df, 'test')