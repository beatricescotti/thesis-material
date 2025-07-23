from ultralytics import YOLO
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(1)})")

# 1. Percorsi dei dataset YAML (ordinati per difficoltà)
datasets = [
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_1/dataset_step_1.yaml',  
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_2/dataset_step_2.yaml',
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_3/dataset_step_3.yaml',
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_4/dataset_step_4.yaml',
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_5/dataset_step_5.yaml',
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_6/dataset_step_6.yaml',
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_7/dataset_step_7.yaml',  
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_8/dataset_step_8.yaml',
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_9/dataset_step_9.yaml',
    '/gwpool/users/bscotti/tesi/dati/final_jet/curriculum/step_10/dataset_step_10.yaml'
]

configs = [
    # Step 1 - 25 jet, no background 
    dict(pretrained = False, freeze =3, dropout=0.3, batch=8, freeze=0, weight_decay=0.01, hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.4, translate = 0.1, scale =0.5, fliplr =0.5),

    # Step 2 - 25 jet, con background 
    dict(pretrained = False, dropout=0.3, batch=8, freeze=5, mosaic=0.1, mixup=0.05, hsv_h=0.005, hsv_s=0.1, translate=0.05, scale=0.1),

    # Step 3 - 12 jet, no background 
    dict(pretrained = False, dropout=0.3, batch=8, freeze=5, mosaic=0.1, mixup=0.05, hsv_h=0.005, hsv_s=0.1, translate=0.05, scale=0.1),

    # Step 4 - 12 jet, con background 
    dict(pretrained = False, dropout=0.3, batch=8, freeze=0, mosaic=0.1, mixup=0.05, hsv_h=0.005, hsv_s=0.1, translate=0.05, scale=0.1),

    # Step 5 - 6 jet, no background
    dict(pretrained = False, dropout=0.25, batch=20, freeze=5, mosaic=0.08, mixup=0.04, hsv_h=0.005, hsv_s=0.08, translate=0.04, scale=0.08),

    # Step 6 - 6 jet, con background
    dict(pretrained = False, dropout=0.25, batch=16, freeze=0, mosaic=0.08, mixup=0.04, hsv_h=0.003, hsv_s=0.07, translate=0.04, scale=0.08),

    # Step 7 - 3 jet, no background 
    dict(pretrained = False, dropout=0.2, batch=16, freeze=5, mosaic=0.05, mixup=0.03, hsv_h=0.002, hsv_s=0.05, translate=0.03, scale=0.05),

    # Step 8 - 3 jet, con background 
    dict(pretrained = False, dropout=0.2, batch=12, freeze=0, mosaic=0.03, mixup=0.02, hsv_h=0.0, hsv_s=0.03, translate=0.02, scale=0.03),

    # Step 9 - 1 jet, no background 
    dict(pretrained = False, dropout=0.15, batch=12, freeze=2, mosaic=0.01, hsv_s=0.01, translate=0.01, scale=0.02),

    # Step 10 - 1 jet, con background 
    dict(pretrained = False, dropout=0.15, batch=8, freeze=0, scale=0.01),
]



# 3. Inizializzazione del modello base
model = YOLO('yolov8m.yaml').to(device)

for i, (yaml_path, cfg) in enumerate(zip(datasets, configs)):
    print(f"\n Step {i+1}/10 - Training on: {yaml_path}")

    # Parametri dinamici in base alla difficoltà 
    if i in range (2,4):  # Steps 1–4 (25-12 jets)
        lr = 5e-5
        box_weight = 7.5
        weight_decay = 0.005
        patience = 15
    elif i < 8:  # Step 5–8 (6-3 jets)
        lr = 1e-5
        box_weight = 5.0
        weight_decay = 0.001
        patience = 10
    else:  # Step 9–10 (1 jet)
        lr = 5e-6
        box_weight = 4.0
        weight_decay = 0.001
        patience = 8

    # Configurazione comune
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
        project='/gwpool/users/bscotti/tesi/dati/final_jet/curriculum_learning_jet_results_false_new',
        name=f'step_{i+1}'
    )

    training_config = {**base_config, **cfg}
    model.train(**training_config)

    # Reload dei pesi migliori per lo step successivo
    model = YOLO(f'/gwpool/users/bscotti/tesi/dati/final_jet/curriculum_learning_jet_results_false_new/step_{i+1}/weights/best.pt').to(device)






    """ dict(dropout=0.4, batch=32, freeze=0),   
    dict(dropout=0.3, batch=16, freeze=10, mosaic=0.1, mixup=0.1, hsv_h=0.005, hsv_s=0.1, translate=0.05, scale=0.1),  # Step 3        # Step 1
    dict(dropout=0.4, batch=32, freeze=5),          # Step 2
    dict(dropout=0.4, batch=16, freeze=0, mosaic=0.1, mixup=0.1, hsv_h=0.005, hsv_s=0.1, translate=0.05, scale=0.1),"""
