import os
import shutil
import pandas as pd
import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split


# modifica i path 
csv_path= '/gwpool/users/bscotti/tesi/dati/immagini_finali_spero/Test_P_Bkg_001b/bbox_P_Bkg_001b.csv'
annotations_df = pd.read_csv(csv_path)
annotations_df['image_name'] = annotations_df['image_name'].str.replace('.npy', '.png')   # se necessario, cambia estensione
annotations_df["image_path"] = "/gwpool/users/bscotti/tesi/dati/immagini_finali_spero/Test_P_Bkg_001b_png/"  + annotations_df["image_name"]

# rimozione delle colonne con le annotazioni di ttbar, che non sono necessarie perchè uso ttbar come fondo 
annotations_df = annotations_df.loc[:, ~annotations_df.columns.str.startswith('ttbar')]
annotations_df = annotations_df.loc[:, ~annotations_df.columns.str.startswith('Unnamed')]
annotations_df = annotations_df.loc[:, ~annotations_df.columns.str.startswith('label')]


# creo un nuoco dataframe in cui rendo più compatte le colonne. 
# dataset finale sarà della forma:      image_name || image_path || x_1 || y_1 || w_1 || h_1 || class_id_1 || x_2 || y_2 || w_2 || h_2 || class_id_2 ...

new_df = pd.DataFrame()
new_df['image_name'] = annotations_df['image_name']
new_df['image_path'] = annotations_df['image_path']

# Numero massimo di box (in teoria è 5, ma ne metto 10 per sicurezza)
num_boxes = 10
for i in range(1, num_boxes + 1):
    hbb_cols = [f'hbb{i}_xmin', f'hbb{i}_ymin', f'hbb{i}_width', f'hbb{i}_height']
    qcd_cols = [f'qcd{i}_xmin', f'qcd{i}_ymin', f'qcd{i}_width', f'qcd{i}_height']
    
    if all(col in annotations_df.columns for col in hbb_cols + qcd_cols):
        x_list, y_list, w_list, h_list, class_list = [], [], [], [], []

        for _, row in annotations_df.iterrows():
            hbb_box = [row[col] for col in hbb_cols]
            qcd_box = [row[col] for col in qcd_cols]

            if not all(np.isnan(hbb_box)):
                x_list.append(hbb_box[0])
                y_list.append(hbb_box[1])
                w_list.append(hbb_box[2])
                h_list.append(hbb_box[3])
                class_list.append(1)
            elif not all(np.isnan(qcd_box)):
                x_list.append(qcd_box[0])
                y_list.append(qcd_box[1])
                w_list.append(qcd_box[2])
                h_list.append(qcd_box[3])
                class_list.append(0)
            else:
                x_list.append(np.nan)
                y_list.append(np.nan)
                w_list.append(np.nan)
                h_list.append(np.nan)
                class_list.append(np.nan)

        new_df[f'x{i}'] = x_list
        new_df[f'y{i}'] = y_list
        new_df[f'w{i}'] = w_list
        new_df[f'h{i}'] = h_list
        new_df[f'class_id{i}'] = class_list


# trasformo in formato YOLO --> (x_center, y_center, width, height) e normalizzo
img_size = (512, 512)  # (width, height)
num_boxes = 5

yolo_data = {
    'image_name': new_df['image_name'],
    'image_path': new_df['image_path']
}

# ciclo su tutti i box con indice i
for i in range(1, num_boxes + 1):
    x_col = f'x{i}'
    y_col = f'y{i}'
    w_col = f'w{i}'
    h_col = f'h{i}'
    class_col = f'class_id{i}'

    x_vals = new_df[x_col]
    y_vals = new_df[y_col]
    w_vals = new_df[w_col]
    h_vals = new_df[h_col]

    # aggiungo le colonne al daraframe finale
    yolo_data[f'class_id{i}'] = new_df[class_col]
    yolo_data[f'x_center{i}'] = (x_vals + w_vals / 2) / img_size[0]
    yolo_data[f'y_center{i}'] = (y_vals + h_vals / 2) / img_size[1]
    yolo_data[f'width{i}']    = w_vals / img_size[0]
    yolo_data[f'height{i}']   = h_vals / img_size[1]

yolo_annotations_df = pd.DataFrame(yolo_data)


# divisione in train/test/validation
train_df, test_valid_df = train_test_split(yolo_annotations_df, train_size=0.8, random_state=43)
test_df, validation_df = train_test_split(test_valid_df, test_size=0.5, random_state=43)

print(f"Dimensione test: {test_df.shape[0]} x {test_df.shape[1]}")
print(f"Dimensione train: {train_df.shape[0]} x {train_df.shape[1]}")
print(f"Dimensione validation: {validation_df.shape[0]} x {validation_df.shape[1]}")




# IMAGES
image_dir = "/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_1/images" 

def move_images(df, split_name):
    destination_path = os.path.join(image_dir, split_name)
    os.makedirs(destination_path, exist_ok=True)

    for _, row in df.iterrows():
        img_path = row["image_path"]  
        img_name = row["image_name"]
        new_path = os.path.join(destination_path, img_name)  
        shutil.copy(img_path, new_path)  

move_images(train_df, "train")
move_images(validation_df, "validation")
move_images(test_df, "test")


# LABELS
def save_annotations(df, output_dir, num_boxes=10):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        lines = []

        for i in range(1, num_boxes + 1):
            class_id = row.get(f'class_id{i}')
            x_center = row.get(f'x_center{i}')
            y_center = row.get(f'y_center{i}')
            width = row.get(f'width{i}')
            height = row.get(f'height{i}')

            # Controlla se TUTTI i valori sono numerici e NON NaN
            if all(pd.notna([class_id, x_center, y_center, width, height])):
                lines.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Se ci sono righe valide, crea e scrivi il file
        if lines:
            filename = os.path.basename(row['image_name']).replace('.png', '.txt')
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))

    print(f"Annotazioni salvate in {output_dir}")

save_annotations(train_df, '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_1/labels/train')
save_annotations(validation_df, '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_1/labels/validation')
save_annotations(test_df, '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_1/labels/test')



# CREAZIONE FILE CONFIGURAZIONE YOLO
yaml_path = "/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_1/dataset.yaml"

# contenuto del file di configurazione
yaml_content = """\
path : /gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_1

train: images/train
val: images/validation
test: images/test 

nc: 2
names: ["qcd", "hbb"] 
"""

with open(yaml_path, "w") as file:
    file.write(yaml_content)
