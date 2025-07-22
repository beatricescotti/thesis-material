import os
import shutil
import pandas as pd
from ultralytics import YOLO
import glob
import numpy as np
import random
import os
import shutil

csv_path_original = '/gwpool/users/bscotti/tesi/csv/annotations_final_resampled_0623.csv'
annotations = pd.read_csv(csv_path_original)

#annotations['image_name'] = annotations['path'].str.split('/').str[-1]  # Estrae il nome dell'immagine dal percorso
#annotations['image_name'] = annotations['image_name'].str.replace('.npy', '.png')

# elimino la colonna path 
#annotations = annotations.drop(columns=['path'], errors='ignore')  
annotations['path'] = '/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/padded_slices_new/' + annotations['img_name']

"""
width, height = 640, 640
yolo_annotations_df = pd.DataFrame({
    'path': annotations['path'],
    'image_name': annotations['image_name'],
    'pid': annotations['pid'],
    'nodule_id': annotations['nodule_id'],
    'slice_id': annotations['slice_id'],
    'class': annotations['is_benign'],
    'x_center': ((annotations['bbox_x1']+ annotations['bbox_x2'])/2) / width,
    'y_center': ((annotations['bbox_y1']+ annotations['bbox_y2'])/2) / height,
    'width': (annotations['bbox_x2']-annotations['bbox_x1']) / width,
    'height': (annotations['bbox_y2']-annotations['bbox_y1']) / height
})

"""

width, height = 704, 704
yolo_annotations_df = pd.DataFrame({
    'path': annotations['path'],
    'dcm_series': annotations['dcm_series'],
    'pid': annotations['pid'],
    'image_name': annotations['img_name'],
    'x_center': (annotations['x']+ annotations['width']/2) / width,
    'y_center': (annotations['y']+ annotations['height']/2) / height,
    'width': annotations['width']/ width,
    'height': annotations['height']/ height
})
# CREAZIONE DATASET TRAIN/VAL E TEST

# Divisione tra Train/Val e Test

# 1. fare una lista univoca dei pid, una riga per ogni pid
unique_pids = yolo_annotations_df['pid'].unique().tolist()
print(unique_pids)
N_tot = len(unique_pids) # numero totale di pid
N = round(N_tot * 0.9) # 80% del numero totale di pid

# 2. estrarre N (che corrisponde all 80% del numero di pid) numeri casuali tra 0 e numero totale di pid
random_indices = random.sample(range(N_tot), N)

# 3. ogni numero casuale estratto corrisponderà a una entry della pid list
selected_pids = [unique_pids[i] for i in random_indices]

# 4. tenere solamente i pid di questa lista per il training a partire dal yolo_annotations_df
train_val_df = yolo_annotations_df[yolo_annotations_df['pid'].isin(selected_pids)]
test_df = yolo_annotations_df[~yolo_annotations_df['pid'].isin(selected_pids)]


# Divisione tra Train e Val  

unique_pids_trv = train_val_df['pid'].unique().tolist()
print(unique_pids_trv)
N_tot_trv = len(unique_pids_trv) # numero totale di pid
N_trv= round(N_tot_trv * 0.9) # 90% del numero totale di pid

random_indices_trv = random.sample(range(N_tot_trv), N_trv)

selected_pids_trv = [unique_pids_trv[i] for i in random_indices_trv]

train_df = train_val_df[train_val_df['pid'].isin(selected_pids_trv)]
validation_df = train_val_df[~train_val_df['pid'].isin(selected_pids_trv)]

# shuffle dei dataframe
validation_df = validation_df.sample(frac=1, random_state=42).reset_index(drop=True) 
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)


# check per vedere se ci sono pid in comune tra train, val e test
# common_pids_trte = train_df['pid'].isin(test_df['pid']).sum()
# common_pids_trva = train_df['pid'].isin(validation_df['pid']).sum()
# common_pids_teva = test_df['pid'].isin(validation_df['pid']).sum()
# print(f"Numero di PID in comune tra test e train: {common_pids_trte}")
# print(f"Numero di PID in comune tra train e val: {common_pids_trva}")
# print(f"Numero di PID in comune tra test e validation: {common_pids_teva}")


""" 
CREAZIONE CARTELLE PER I DATASET
i dati devono essere organizzati nel seguente modo per essere allenati con YOLO:   
 - dataset 
     dataset.yaml                  file di configurazione
     - images                      sono le immagini in png divise in base ai dataset 
         - train
         - validation  
         - test
     - labels                      sono file .txt con le coordinate dei bounding box, uno per ogni immagine
         - train
         - validation
         - test
"""

# IMAGES
image_dir = "/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_NLST_padded/images" # sostituire con il path di destinazione immagini

def move_images(df, split_name):
    # creo la cartella di destinazione se non esiste
    destination_path = os.path.join(image_dir, split_name)
    os.makedirs(destination_path, exist_ok=True)

    for _, row in df.iterrows():
        img_path = row["path"]  # percorso dell'immagine
        img_name = os.path.basename(img_path)  # estrazione del nome del file
        new_path = os.path.join(destination_path, img_name)  # nuovo percorso
        shutil.copy(img_path, new_path)  # copia l'immagine nella cartella di destinazione

# Applicazione della funzione per train, validation e test
move_images(train_df, "train")
move_images(validation_df, "validation")
move_images(test_df, "test")


# LABELS
def save_annotations(df, output_dir): # crea la cartella
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        # estraggo il nome del file dal path e metto .txt
        filename = os.path.basename(row['path']).replace('.png', '.txt')

        filepath = os.path.join(output_dir, filename) # percorso dove salvare il file

        # informazioni da mettere nel file
        #class_id = row['class']
        class_id = 0
        x_center = row['x_center']
        y_center = row['y_center']
        width = row['width']
        height = row['height'] 

        with open(filepath, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Annotazioni salvate in {output_dir}")


save_annotations(train_df, '/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_NLST_padded/labels/train')
save_annotations(validation_df, '/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_NLST_padded/labels/validation')
save_annotations(test_df, '/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_NLST_padded/labels/test')



# CREAZIONE FILE CONFIGURAZIONE YOLO
yaml_path = "/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_NLST_padded/dataset.yaml"

# contenuto del file di configurazione
yaml_content = """\
path : /gwpool/users/bscotti/tesi/dati/dati_medici/dataset_NLST_padded

train: images/train
val: images/validation
test: images/test 

nc: 1  
names: ["tumore"]
"""

with open(yaml_path, "w") as file:
    file.write(yaml_content)