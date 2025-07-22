from ultralytics import YOLO
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(1)})")

# TRAINING
model = YOLO('/gwpool/users/bscotti/tesi/dati/final_jet/curriculum_learning_jet_results/step_10/weights/best.pt').to(device)

# Training con svuotamento della memoria
model.train(
    data='/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_andre/dataset_andre/dataset_padded/dataset.yaml',
    epochs=100,
    imgsz=736,
    batch=8,
    optimizer='SGD',
    lr0=1e-4,
    cos_lr=True,
    patience=10,
    weight_decay=0.01,
    project='/gwpool/users/bscotti/tesi/dati/final_jet/training_curriculum_medical',
    name='DLCS_step10',
    box=7.5,
    cls=1,
    dfl=1.5,
    dropout=0.3, 
    degrees=10,
    flipud=0.3,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

# Dopo ogni epoca, svuotare la memoria GPU
for epoch in range(100):  # O il numero di epoche che stai usando
    # Puoi aggiungere il codice per l'allenamento di ogni epoca qui
    # ...
    
    # Svuotare la memoria GPU
    torch.cuda.empty_cache()
    print(f"Memory cleared after epoch {epoch + 1}")