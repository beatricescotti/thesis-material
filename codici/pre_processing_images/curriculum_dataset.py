import os
import random
import shutil
import yaml
from pathlib import Path

def create_yaml_file(step_dir, step_num, class_names):
    """Crea il file YAML di configurazione per YOLO"""
    yaml_content = {
        'path': str(Path(step_dir).absolute()),
        'train': 'images/train',
        'val': 'images/validation',
        'test': 'images/test',
        'names': class_names
    }
    
    yaml_path = Path(step_dir) / f"dataset_step_{step_num}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"Created YAML config at: {yaml_path}")
    return yaml_path

def copy_dataset_structure(source_dir, dest_dir):
    """Copia l'intera struttura del dataset mantenendo:
    - images/train, images/validation, images/test
    - labels/train, labels/validation, labels/test
    """
    # Crea struttura cartelle
    (dest_dir / 'images').mkdir(exist_ok=True)
    (dest_dir / 'labels').mkdir(exist_ok=True)
    
    for split in ['train', 'validation', 'test']:
        # Copia immagini
        src_images = Path(source_dir) / 'images' / split
        if src_images.exists():
            (dest_dir / 'images' / split).mkdir(exist_ok=True)
            for img_path in src_images.glob('*'):
                if img_path.is_file():
                    shutil.copy(img_path, dest_dir / 'images' / split)
        
        # Copia labels
        src_labels = Path(source_dir) / 'labels' / split
        if src_labels.exists():
            (dest_dir / 'labels' / split).mkdir(exist_ok=True)
            for label_path in src_labels.glob('*.txt'):
                shutil.copy(label_path, dest_dir / 'labels' / split)
            # Crea file vuoti per immagini senza annotazioni
            for img_path in (dest_dir / 'images' / split).glob('*'):
                label_name = f"{img_path.stem}.txt"
                dest_label = dest_dir / 'labels' / split / label_name
                if not dest_label.exists():
                    dest_label.touch()

def add_previous_data(current_step_dir, prev_step_dir, percentage):
    """Aggiunge una percentuale dei dati precedenti"""
    added_counts = {'train': 0, 'validation': 0, 'test': 0}
    
    for split in ['train', 'validation', 'test']:
        # Immagini dal passo precedente
        prev_images = list((prev_step_dir / 'images' / split).glob('*'))
        if not prev_images:
            continue
            
        n_images = max(1, int(len(prev_images) * percentage))
        selected_images = random.sample(prev_images, n_images) if n_images < len(prev_images) else prev_images
        
        # Copia immagini e labels
        for img_path in selected_images:
            # Copia immagine
            shutil.copy(img_path, current_step_dir / 'images' / split)
            
            # Copia label
            label_name = f"{img_path.stem}.txt"
            src_label = prev_step_dir / 'labels' / split / label_name
            dest_label = current_step_dir / 'labels' / split / label_name
            
            if src_label.exists():
                shutil.copy(src_label, dest_label)
            else:
                dest_label.touch()
            
            added_counts[split] += 1
    
    return added_counts

def create_curriculum_dataset(base_datasets, output_dir="/gwpool/users/bscotti/tesi/dati/final_jet/curriculum", class_names=None, prev_percentage=0.2):
    """
    Crea i dataset per curriculum learning con la struttura esatta richiesta:
    step_X/
    ├── images/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    └── dataset_step_X.yaml
    """
    if class_names is None:
        class_names = {0: 'qcd', 1: 'hbb'}
    
    Path(output_dir).mkdir(exist_ok=True)
    yaml_paths = []
    
    for step_num, current_ds in enumerate(base_datasets, start=1):
        print(f"\n=== Processing step {step_num}: {Path(current_ds).name} ===")
        
        # Crea cartella per questo step
        step_dir = Path(output_dir) / f"step_{step_num}"
        step_dir.mkdir(exist_ok=True)
        
        # 1. Copia la struttura completa del dataset corrente
        print("Copying current dataset structure...")
        copy_dataset_structure(current_ds, step_dir)
        
        # 2. Aggiungi dati dagli step precedenti a TUTTE le split
        if step_num > 1:
            print(f"\nAdding {prev_percentage*100}% of previous steps data...")
            for prev_step in range(1, step_num):
                prev_ds = Path(output_dir) / f"step_{prev_step}"
                added = add_previous_data(step_dir, prev_ds, prev_percentage)
                print(f"Added from step {prev_step}: {added['train']} train, {added['validation']} val, {added['test']} test")
        
        # 3. Crea il file YAML
        yaml_path = create_yaml_file(step_dir, step_num, class_names)
        yaml_paths.append(yaml_path)
        
        # Report
        print("\nCurrent counts:")
        for split in ['train', 'validation', 'test']:
            img_count = len(list((step_dir / 'images' / split).glob('*')))
            lbl_count = len(list((step_dir / 'labels' / split).glob('*.txt')))
            print(f"{split.upper()}: {img_count} images, {lbl_count} labels")
    
    return yaml_paths

# Configurazione
if __name__ == "__main__":
    datasets_ordered = [
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_no_bkg_25",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_bkg_25",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_no_bkg_12",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_bkg_12",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_no_bkg_6",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_bkg_6",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_no_bkg_3",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_bkg_3",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_no_bkg_1",
        "/gwpool/users/bscotti/tesi/dati/final_jet/dataset_poisson_bkg_1"
    ]
    # Verifica preliminare
    print("=== Verifica preliminare dei dataset ===")
    for ds in datasets_ordered:
        ds_path = Path(ds)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset non trovato: {ds}")
        
        print(f"\nDataset: {ds_path.name}")
        for split in ['train', 'validation', 'test']:
            img_dir = ds_path / 'images' / split
            lbl_dir = ds_path / 'labels' / split
            print(f"{split.upper()}:")
    
    # Esegui la creazione
    print("\n=== Inizio creazione curriculum ===")
    yaml_files = create_curriculum_dataset(
        datasets_ordered,
        class_names={0: "qcd", 1: "hbb"},
        prev_percentage=0.2  # 20% dei dati precedenti
    )
    
    print("\n=== Comandi per il training ===")
    for i, yaml_path in enumerate(yaml_files, start=1):
        print(f"STEP {i}: yolo train data={yaml_path} ...")