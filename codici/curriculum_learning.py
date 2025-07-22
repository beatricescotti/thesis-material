from ultralytics import YOLO
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# 1. Percorsi dei dataset YAML (ordinati per difficoltà)
datasets = [
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_nuovo_25/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_25/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_nuovo_12/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_12/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_nuovo_6/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_6/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_nuovo_3/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_3/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_nuovo_1/dataset.yaml',
    '/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_1/dataset.yaml'
]

# 2. Configurazioni per ogni step
configs = [
    dict(dropout=0.4, batch=32, freeze=0),
    dict(dropout=0.4, batch=32, freeze=10),
    dict(dropout=0.3, batch=24, freeze=5, mosaic=0.1, mixup=0.05, hsv_h=0.005, hsv_s=0.1, translate=0.05, scale=0.1),
    dict(dropout=0.3, batch=20, freeze=10, mosaic=0.15, mixup=0.1, hsv_h=0.01, hsv_s=0.15, translate=0.1, scale=0.15),
    dict(dropout=0.3, batch=24, freeze=5, mosaic=0.1, mixup=0.05, hsv_h=0.005, hsv_s=0.1, translate=0.05, scale=0.1),
    dict(dropout=0.3, batch=20, freeze=10, mosaic=0.15, mixup=0.1, hsv_h=0.01, hsv_s=0.15, translate=0.1, scale=0.15),
    dict(dropout=0.2, batch=20, freeze=5, mosaic=0.15, mixup=0.1, hsv_h=0.01, hsv_s=0.15, translate=0.1, scale=0.15),
    dict(dropout=0.2, batch=16, freeze=5, mosaic=0.2, mixup=0.15, hsv_h=0.015, hsv_s=0.2, translate=0.15, scale=0.2),
    dict(dropout=0.15, batch=12, freeze=0, mosaic=0.25, mixup=0.2, hsv_h=0.02, hsv_s=0.25, translate=0.2, scale=0.3),
    dict(dropout=0.15, batch=8, freeze=0, mosaic=0.3, mixup=0.2, hsv_h=0.025, hsv_s=0.3, translate=0.25, scale=0.35)
]

# 3. Inizializzazione del modello base
model = YOLO('yolov8s.pt').to(device)

for i, (yaml_path, cfg) in enumerate(zip(datasets, configs)):
    print(f"\n Step {i+1}/10 - Training on: {yaml_path}")

    # Modifica dinamica dei parametri
    lr = 1e-4 if i < 4 else 5e-5
    weight_decay = 0.001 if i < 6 else 5e-4
    box_weight = 7.5 if i < 6 else 5.0
    patience = 15 if i < 6 else 10

    base_config = dict(
        epochs=100,
        data=yaml_path,
        imgsz=512,
        lr0=lr,
        cos_lr=True,
        optimizer='AdamW',
        weight_decay=weight_decay,
        patience=patience,
        box=box_weight,
        cls=1.0,
        dfl=1.5,
        project='/gwpool/users/bscotti/tesi/curriculum_learning_7',
        name=f'step_{i+1}'
    )

    # Unione base + config specifica
    training_config = {**base_config, **cfg}

    # Training
    model.train(**training_config)

    # Aggiornamento con i pesi migliori
    best_model_path = f'/gwpool/users/bscotti/tesi/curriculum_learning_7/step_{i+1}/weights/best.pt'
    model = YOLO(best_model_path).to(device)