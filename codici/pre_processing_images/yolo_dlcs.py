import os
import shutil
import pandas as pd
import random

# SCRIPT CHE CREA I DATASET IN FORMATO YOLO PER OGNI STEP DEL CURRICULUM MANTENENDO LO STESSO VALIDATION E TEST SET 
# LA DIVISIONE AVVIENE SOLO A LIVELLO DI TRAIN 

csv_path = '/gwpool/users/bscotti/tesi/2d_annotations_pruned_threshold_sorted.csv' # già ordinato in ordine decrescente per nodule intensity
base_output = '/gwpool/users/bscotti/tesi/prova_curriculum'
img_width, img_height = 736, 736

df = pd.read_csv(csv_path)
print(f"Totale immagini: {len(df)}, totale pazienti: {df['pid'].nunique()}")


def move_images(df, destination_base, split_name):
    """
    SPOSTA LE IMMAGINI NELLA CARTELLA DESTINAZIONE 
    """
    dest = os.path.join(destination_base, 'images', split_name)
    os.makedirs(dest, exist_ok=True)
    for _, row in df.iterrows():
        shutil.copy(row['path'], os.path.join(dest, os.path.basename(row['path'])))


def save_annotations(df, destination_base, split_name):
    """
    SALVA LE ANNOTAZIONI IN FORMATO YOLO (normalizzate e in formato .txt)
    """
    dest = os.path.join(destination_base, 'labels', split_name)
    os.makedirs(dest, exist_ok=True)
    for _, row in df.iterrows():
        filename = os.path.basename(row['path']).replace('.png', '.txt')
        filepath = os.path.join(dest, filename)
        class_id = row['is_benign']
        x_center = ((row['bbox_x1'] + row['bbox_x2']) / 2) / img_width
        y_center = ((row['bbox_y1'] + row['bbox_y2']) / 2) / img_height
        width = (row['bbox_x2'] - row['bbox_x1']) / img_width
        height = (row['bbox_y2'] - row['bbox_y1']) / img_height
        with open(filepath, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def create_yolo_dataset(train_df, val_df, test_df, output_path):
    """
    CREA IL DATASET IN FORMATO YOLO 
    sposta le immagini 
    crea le annotazioni 
    crea il file di configurazione 
    """
    for split_name, split_df in zip(['train', 'validation', 'test'],
                                    [train_df, val_df, test_df]):
        move_images(split_df, output_path, split_name)
        save_annotations(split_df, output_path, split_name)

    yaml_path = os.path.join(output_path, 'dataset.yaml')
    yaml_content = f"""path: {output_path}
train: images/train
val: images/validation
test: images/test
nc: 2
names: ["maligno", "benigno"]
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)



def split_global(input_df, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    DIVIDE IL DATAFRAME IN TRE PARTI:
    - train: 80% dei pazienti 
    - validation: 10% dei pazienti
    - test: 10% dei pazienti
    validation e test saranno gli stessi per tutti gli step, il training dovrà essere suvddiviso in tre step
    """
    unique_pids = input_df['pid'].unique().tolist()
    random.Random(seed).shuffle(unique_pids)

    n_total = len(unique_pids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_pids = unique_pids[:n_train]
    val_pids = unique_pids[n_train:n_train+n_val]
    test_pids = unique_pids[n_train+n_val:]

    return (
        input_df[input_df['pid'].isin(train_pids)],
        input_df[input_df['pid'].isin(val_pids)],
        input_df[input_df['pid'].isin(test_pids)],
    )

global_train_df, global_val_df, global_test_df = split_global(df)
print(f"Split globale PID: train={global_train_df['pid'].nunique()}, "
      f"val={global_val_df['pid'].nunique()}, test={global_test_df['pid'].nunique()}")



ordered_pids = global_train_df['pid'].drop_duplicates().tolist()
n_pids = len(ordered_pids)

# divisione in tre step basata sui PID 
steps_pids = {
    "step1": ordered_pids[:int(0.33*n_pids)],
    "step2": ordered_pids[:int(0.66*n_pids)],
    "step3": ordered_pids
}

for step_name, pids in steps_pids.items():
    step_train_df = global_train_df[global_train_df['pid'].isin(pids)]
    print(f"\nProcessing {step_name}: {len(step_train_df)} train images, "
          f"Val={len(global_val_df)}, Test={len(global_test_df)}")

    output_path = os.path.join(base_output, step_name)
    create_yolo_dataset(step_train_df, global_val_df, global_test_df, output_path)
    print(f"Dataset {step_name} creato in {output_path}")