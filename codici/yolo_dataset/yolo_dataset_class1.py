import os
import shutil
import pandas as pd
import glob
import random
from sklearn.model_selection import train_test_split


#  CLASSE 1 CONTIENE 2 OGGETTI DI TIPO QCD e 1 OGGETTO DI TIPO HBB

# lettura del file CSV
csv_path= '/gwpool/users/bscotti/tesi/dati/dati_jet/immagini_finali_spero/fixed_patch/final_1_noP_noBkg_025/bbox_1_noP_noBkg_025.csv'
annotations_df = pd.read_csv(csv_path)
annotations_df['image_name'] = annotations_df['image_name'].str.replace('.npy', '.png') 
annotations_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/dati_jet/immagini_finali_spero/fixed_patch/final_1_noP_noBkg_025_png/" + annotations_df["image_name"]



# dataset in formato yolo (x_centro, y_centro, width, height) con valori normalizzati tra 0 e 1 e non in formato (x, y, width, height)
# CLASSE 1 

img_size = (512, 512, 1)
yolo_annotations_df = pd.DataFrame({
    'image_name' : annotations_df['image_name'],
    'image_path': annotations_df['image_path'],

    'class_id1': 1,
    'x_center1': (annotations_df['hbb_xmin']+ annotations_df['hbb_width']/2) / img_size[0],
    'y_center1': (annotations_df['hbb_ymin'] + annotations_df['hbb_height']/2) / img_size[1],
    'width1': annotations_df['hbb_width'] / img_size[0],
    'height1': annotations_df['hbb_height'] / img_size[1],

    'class_id2' : 0,
    'x_center2': (annotations_df['qcd2_xmin']+ annotations_df['qcd2_width']/2) / img_size[0],
    'y_center2': (annotations_df['qcd2_ymin'] + annotations_df['qcd2_height']/2) / img_size[1],
    'width2': annotations_df['qcd2_width'] / img_size[0],
    'height2': annotations_df['qcd2_height'] / img_size[1],

    'class_id3': 0,
    'x_center3': (annotations_df['qcd1_xmin']+ annotations_df['qcd1_width']/2) / img_size[0],
    'y_center3': (annotations_df['qcd1_ymin'] + annotations_df['qcd1_height']/2) / img_size[1],
    'width3': annotations_df['qcd1_width'] / img_size[0],
    'height3': annotations_df['qcd1_height'] / img_size[1]
})


# Divisione in train, test e validation
# 80% training, 10% validation, 10% test

train_df, test_valid_df = train_test_split(yolo_annotations_df, train_size=0.8, random_state=43)
test_df, validation_df = train_test_split(test_valid_df, test_size=0.5, random_state=43)

print(f"Dimensione test: {test_df.shape[0]} x {test_df.shape[1]}")
print(f"Dimensione train: {train_df.shape[0]} x {train_df.shape[1]}")
print(f"Dimensione validation: {validation_df.shape[0]} x {validation_df.shape[1]}")



# IMAGES
# Copia delle immagini nelle rispettive cartelle di destinazione per il training con yolo 
image_dir = "/gwpool/users/bscotti/tesi/dati/dataset_nuovo_25/images" # sostituire con il path di destinazione immagini

def move_images(df, split_name):
    destination_path = os.path.join(image_dir, split_name)
    os.makedirs(destination_path, exist_ok=True)

    for _, row in df.iterrows():
        img_path = row["image_path"] 
        img_name = row["image_name"]
        new_path = os.path.join(destination_path, img_name)  
        shutil.copy(img_path, new_path)  

# Applicazione della funzione per train, validation e test
move_images(train_df, "train")
move_images(validation_df, "validation")
move_images(test_df, "test")



# LABELS
# Creazione delle cartelle per le etichette
def save_annotations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        filename = os.path.basename(row['image_name']).replace('.png', '.txt')
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


save_annotations(train_df, '/gwpool/users/bscotti/tesi/dati/dataset_nuovo_25/labels/train')
save_annotations(validation_df, '/gwpool/users/bscotti/tesi/dati/dataset_nuovo_25/labels/validation')
save_annotations(test_df, '/gwpool/users/bscotti/tesi/dati/dataset_nuovo_25/labels/test')



# CREAZIONE FILE CONFIGURAZIONE YOLO
yaml_path = "/gwpool/users/bscotti/tesi/dati/dataset_nuovo_25/dataset.yaml"

# contenuto del file di configurazione
yaml_content = """\
path : /gwpool/users/bscotti/tesi/dati/dataset_nuovo_25

train: images/train
val: images/validation
test: images/test 

nc: 2
names: ["qcd", "hbb"] 
"""

with open(yaml_path, "w") as file:
    file.write(yaml_content)