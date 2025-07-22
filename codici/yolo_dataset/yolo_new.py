import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


csv_path = '/gwpool/users/bscotti/tesi/dati/final_jet/patches/bbox_P_Bkg_012_labeled.csv'
df = pd.read_csv(csv_path)
df['image_name'] = df['img_name'].str.replace('.npy', '.png', regex=False)
df['path'] = '/gwpool/users/bscotti/tesi/dati/final_jet/patches/P_Bkg_012_png/' + df['image_name']
df['label_blob_index'] = df['label_blob'].map({'qcd': 0, 'hbb': 1, 'ttbar': 2})
df = df[df['label_blob_index'] != 2].reset_index(drop=True) # esclude le immagini con label 2 (jet)

# separa le immagini in train, validation e test (uniche per image_name)
unique_images = df[['image_name', 'path']].drop_duplicates()
train_imgs, testval_imgs = train_test_split(unique_images, train_size=0.8, random_state=42)
val_imgs, test_imgs = train_test_split(testval_imgs, test_size=0.5, random_state=42)

split_map = {}
for name in train_imgs['image_name']: split_map[name] = 'train'
for name in val_imgs['image_name']: split_map[name] = 'validation'
for name in test_imgs['image_name']: split_map[name] = 'test'

# costanti
output_base = '/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_bkg_12'
img_dir = os.path.join(output_base, 'images')
label_dir = os.path.join(output_base, 'labels')
img_size = (512, 512)

# creazione delle cartelle
for split in ['train', 'validation', 'test']:
    os.makedirs(os.path.join(img_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_dir, split), exist_ok=True)

# genero annotazioni
grouped = df.groupby('image_name')

for image_name, group in grouped:
    split = split_map.get(image_name)
    if split is None:
        continue

    image_path = group['path'].iloc[0]
    shutil.copy(image_path, os.path.join(img_dir, split, image_name))

    lines = []

    for _, row in group.iterrows():
        # Se i dati sono incompleti, salta
        if pd.isna(row['label_blob_index']) or pd.isna(row['x_min']) or pd.isna(row['y_min']) or pd.isna(row['w']) or pd.isna(row['h']):
            continue

        class_id = int(row['label_blob_index'])
        x_center = (row['x_min'] + row['w'] / 2) / img_size[0]
        y_center = (row['y_min'] + row['h'] / 2) / img_size[1]
        width = row['w'] / img_size[0]
        height = row['h'] / img_size[1]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Scrivi SEMPRE un file, anche se vuoto (YOLO accetta file vuoti per immagini di background)
    label_file = image_name.replace('.png', '.txt')
    with open(os.path.join(label_dir, split, label_file), 'w') as f:
        f.write('\n'.join(lines))

# file di configurazione 
yaml_path = os.path.join(output_base, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"""\
path: {output_base}

train: images/train
val: images/validation
test: images/test

nc: 2
names: ["qcd", "hbb"]
""")

print(f"✔️ YAML creato in: {yaml_path}")