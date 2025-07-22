import os
import shutil
import pandas as pd
from ultralytics import YOLO
import glob
import numpy as np
import random
import os
import shutil


# importo il file csv
csv_path = '/gwpool/users/bscotti/tesi/csv/annotations.csv'
annotations_df = pd.read_csv(csv_path)

# qui riduco la dimensione del file annotations in modo tale da avere solo le righe che corrispondono alle immagini che ho nella cartella slice_filtrate_batch1
images_path = '/gwpool/users/bscotti/tesi/dati/slices_white' # totale : 8840 immagini

nomi_immagini = [] # creazione di una lista di immagini che verrà riempita con i nomi delle immagini a cui tolgo l'estensione perchè non ci sono dentro al csv
for file in os.listdir(images_path):
    nome_immagine_senza_estensione = os.path.splitext(file)[0]  # Rimuovi l'estensione
    nomi_immagini.append(nome_immagine_senza_estensione)

righe_filtrate = [] # creo un'altra lista in cui tengo solamente le righe il cui nome = nome di un'immagine nella lista sopra
for index, row in annotations_df.iterrows():
    nome_immagine_csv = row.iloc[2]  # estraggo il nome dell'immagine dalla prima colonna del csv
    if nome_immagine_csv in nomi_immagini:
        righe_filtrate.append(row)

# creazione di un nuovo dataframe che ha come righe solo quelle per cui ha trovato la corrispondenza tra le due liste
filtered_annotations_df = pd.DataFrame(righe_filtrate)


# dataset in formato yolo (x_centro, y_centro, width, height) con valori normalizzati tra 0 e 1 e non in formato (x, y, width, height)
img_size = (512, 512, 1)
yolo_annotations_df = pd.DataFrame({
    'pid': filtered_annotations_df['pid'],
    'dcm_series' : filtered_annotations_df['dcm_series'],
    'uid_slice' : filtered_annotations_df['uid_slice'],
    'x_center': (filtered_annotations_df['x']+filtered_annotations_df['width']/2) / img_size[0],
    'y_center': (filtered_annotations_df['y'] + filtered_annotations_df['height']/2) / img_size[1],
    'width': filtered_annotations_df['width'] / img_size[0],
    'height': filtered_annotations_df['height'] / img_size[1]
})

# inserisco il path completo delle immagini
yolo_annotations_df['uid_slice'] = yolo_annotations_df['uid_slice'].apply(lambda x: os.path.join(images_path, f"{x}.png" if not x.endswith('.png') else x))



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

# tolgo le colonne pid e dcm_series
train_df = train_df[['uid_slice', 'x_center', 'y_center', 'width', 'height']] 
test_df = test_df[['uid_slice', 'x_center', 'y_center', 'width', 'height']]
validation_df = validation_df[['uid_slice', 'x_center', 'y_center', 'width', 'height']]


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
image_dir = "/gwpool/users/bscotti/tesi/dati/dataset/images" # sostituire con il path di destinazione immagini

def move_images(df, split_name):
    # creo la cartella di destinazione se non esiste
    destination_path = os.path.join(image_dir, split_name)
    os.makedirs(destination_path, exist_ok=True)

    for _, row in df.iterrows():
        img_path = row["uid_slice"]  # percorso dell'immagine
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
        filename = os.path.basename(row['uid_slice']).replace('.png', '.txt')

        filepath = os.path.join(output_dir, filename) # percorso dove salvare il file

        # informazioni da mettere nel file
        x_center = row['x_center']
        y_center = row['y_center']
        width = row['width']
        height = row['height']

        class_id = 0  # dal momento che devo fare una regressione, ho una sola classe (il boundingbox). va messo per forza perchè yolo fa anche da classificatore

        with open(filepath, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Annotazioni salvate in {output_dir}")


save_annotations(train_df, '/gwpool/users/bscotti/tesi/dati/dataset/labels/train')
save_annotations(validation_df, '/gwpool/users/bscotti/tesi/dati/dataset/labels/validation')
save_annotations(test_df, '/gwpool/users/bscotti/tesi/dati/dataset/labels/test')



# CREAZIONE FILE CONFIGURAZIONE YOLO
yaml_path = "/gwpool/users/bscotti/tesi/dati/dataset/dataset.yaml"

# contenuto del file di configurazione
yaml_content = """\
path : /gwpool/users/bscotti/tesi/dati/dataset

train: images/train
val: images/validation
test: images/test 

nc: 1  
names: ["boundingbox"] 
"""

with open(yaml_path, "w") as file:
    file.write(yaml_content)