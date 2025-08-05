import os
import shutil
from PIL import Image
import pandas as pd

def move_images(df, destination_base, split_name):
    dest = os.path.join(destination_base, 'images', split_name)
    os.makedirs(dest, exist_ok=True)
    copied = set()
    for img_path in df['path'].unique():
        img_name = os.path.basename(img_path)
        if img_path not in copied:
            shutil.copy(img_path, os.path.join(dest, img_name))
            copied.add(img_path)

def save_annotations(df, destination_base, split_name):
    dest = os.path.join(destination_base, 'labels', split_name)
    os.makedirs(dest, exist_ok=True)

    grouped = df.groupby('path')

    for img_path, group in grouped:
        img_name = os.path.basename(img_path)
        filename = os.path.splitext(img_name)[0] + '.txt'
        filepath = os.path.join(dest, filename)

        with Image.open(img_path) as img:
            img_width, img_height = img.size

        with open(filepath, 'w') as f:
            for _, row in group.iterrows():
                class_id = row['is_benign']  # modifica se hai più classi
                x_center = ((row['bbox_x1'] + row['bbox_x2']) / 2) / img_width
                y_center = ((row['bbox_y1'] + row['bbox_y2']) / 2) / img_height
                width = (row['bbox_x2'] - row['bbox_x1']) / img_width
                height = (row['bbox_y2'] - row['bbox_y1']) / img_height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_yolo_dataset(train_csv, val_csv, test_csv, output_path):
    print("Caricamento CSV...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print("Sposto immagini train e creo annotazioni...")
    move_images(train_df, output_path, 'train')
    save_annotations(train_df, output_path, 'train')

    print("Sposto immagini validation e creo annotazioni...")
    move_images(val_df, output_path, 'validation')
    save_annotations(val_df, output_path, 'validation')

    print("Sposto immagini test e creo annotazioni...")
    move_images(test_df, output_path, 'test')
    save_annotations(test_df, output_path, 'test')

    # Creo dataset.yaml
    yaml_path = os.path.join(output_path, 'dataset.yaml')
    yaml_content = f"""\
path: {output_path}

train: images/train
val: images/validation
test: images/test

nc: 2
names: ["maligno", "benigno"]
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"File dataset.yaml creato in {yaml_path}")

# --- esempio d'uso ---

if __name__ == "__main__":
    train_csv = '/gwpool/users/bscotti/tesi/train_step3.csv'  # cambia con il csv train step che vuoi usare
    val_csv = '/gwpool/users/bscotti/tesi/valid.csv'
    test_csv = '/gwpool/users/bscotti/tesi/test.csv'
    output_path = '/gwpool/users/bscotti/tesi/step3'
    os.makedirs(output_path, exist_ok=True)

    create_yolo_dataset(train_csv, val_csv, test_csv, output_path)
