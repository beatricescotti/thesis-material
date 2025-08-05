import os
import shutil
import pandas as pd
import numpy as np
import random


csv_path = '/gwpool/users/bscotti/tesi/2d_annotations_pruned_threshold_sorted.csv'
base_output = '/gwpool/users/bscotti/tesi/prova_curriculum_2'
img_width, img_height = 736, 736
df = pd.read_csv(csv_path)
print(f"Totale immagini: {len(df)}, totale pazienti: {df['pid'].nunique()}")


def split_pids(pids, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.Random(seed).shuffle(pids)
    n_total = len(pids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    train_pids = pids[:n_train]
    val_pids = pids[n_train:n_train + n_val]
    test_pids = pids[n_train + n_val:]
    return train_pids, val_pids, test_pids

def get_split_df(df, train_pids, val_pids, test_pids):
    train_df = df[df['pid'].isin(train_pids)]
    val_df = df[df['pid'].isin(val_pids)]
    test_df = df[df['pid'].isin(test_pids)]
    return train_df, val_df, test_df

def move_images(df, destination_base, split_name):
    dest = os.path.join(destination_base, 'images', split_name)
    os.makedirs(dest, exist_ok=True)
    for _, row in df.iterrows():
        img_path = row['path']
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dest, img_name))

def save_annotations(df, destination_base, split_name):
    dest = os.path.join(destination_base, 'labels', split_name)
    os.makedirs(dest, exist_ok=True)
    for _, row in df.iterrows():
        filename = os.path.basename(row['path']).replace('.png', '.txt')
        filepath = os.path.join(dest, filename)
        class_id = 0
        x_center = ((row['bbox_x1'] + row['bbox_x2']) / 2) / img_width
        y_center = ((row['bbox_y1'] + row['bbox_y2']) / 2) / img_height
        width = (row['bbox_x2'] - row['bbox_x1']) / img_width
        height = (row['bbox_y2'] - row['bbox_y1']) / img_height
        with open(filepath, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def create_yolo_dataset(train_df, val_df, test_df, output_path):
    for split_name, split_df in zip(['train', 'validation', 'test'], [train_df, val_df, test_df]):
        move_images(split_df, output_path, split_name)
        save_annotations(split_df, output_path, split_name)

    yaml_path = os.path.join(output_path, 'dataset.yaml')
    yaml_content = f"""
path: {output_path}

train: images/train
val: images/validation
test: images/test

nc: 2
names: ["maligno", "benigno"]
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

# === ORDINA I PID ===
ordered_pids = df['pid'].drop_duplicates().tolist()
n_total = len(ordered_pids)

steps = {
    "step1": ordered_pids[:int(0.33 * n_total)],
    "step2": ordered_pids[:int(0.66 * n_total)],
    "step3": ordered_pids
}

# === CICLO SUI 3 STEP ===
for step_name, pids in steps.items():
    print(f"\nProcessing {step_name}...")
    step_df = df[df['pid'].isin(pids)]

    train_pids, val_pids, test_pids = split_pids(pids, train_ratio=0.8, val_ratio=0.1, seed=42)
    train_df, val_df, test_df = get_split_df(step_df, train_pids, val_pids, test_pids)

    output_path = os.path.join(base_output, step_name)
    create_yolo_dataset(train_df, val_df, test_df, output_path)

    print(f"Dataset {step_name} creato con {len(train_df)} train, {len(val_df)} val, {len(test_df)} test immagini.")
