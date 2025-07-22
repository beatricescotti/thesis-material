from ultralytics import YOLO
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parto ad allenare dallo step 5 del curriculum learning
model = YOLO('yolov8m.pt').to(device)
# model = YOLO('yolov8s.pt').to(device)  

model.train(
    data='/gwpool/users/bscotti/tesi/dati/dataset_poisson_bkg_nuovo_3/dataset.yaml',  
    epochs=100,
    imgsz=512,
    batch=12,
    optimizer='AdamW',
    lr0=5e-5,              
    weight_decay=5e-4,
    dropout=0.2,           
    patience=10,           
    cos_lr=True,
    box=5.0,
    cls=1.0,
    dfl=1.5,
    mosaic=0.03,            
    mixup=0.02,
    hsv_h=0.0,
    hsv_s=0.03,
    translate=0.02,
    scale=0.03,
    project='/gwpool/users/bscotti/tesi/train_poisson/3_poisson_fondo/dataset_nuovo',
    name='confronto_curr'
)
