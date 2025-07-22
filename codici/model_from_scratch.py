import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt


# Verifica se la GPU è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

# Percorsi ai dati
image_path = '/gwpool/users/bscotti/tesi/dati/folder_1_ttbar_qcd_hbb_100_jet'
csv_path = '/gwpool/users/bscotti/tesi/csv/bounding_boxes_hbb_100_jet.csv'

# Carica il dataframe
annotations_df = pd.read_csv(csv_path)
annotations_df["image_path"] = image_path + '/' + annotations_df["image_name"]

# Suddivisione in train, validation e test
from sklearn.model_selection import train_test_split
train_df, test_valid_df = train_test_split(annotations_df, train_size=0.75, random_state=43)
test_df, validation_df = train_test_split(test_valid_df, test_size=0.5, random_state=43)

# Dataset personalizzato
class BoundingBoxDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['image_path']).convert('L')
        bbox = np.array([row['bbox_x_min'], row['bbox_y_min'], row['bbox_x_min'] + row['bbox_width'], row['bbox_y_min'] + row['bbox_height']], dtype=np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(bbox)

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Creazione dei DataLoader
batch_size = 8
train_dataset = BoundingBoxDataset(train_df, transform=transform)
val_dataset = BoundingBoxDataset(validation_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Definizione del modello CNN
class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.4),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.4)
        )
        
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 4)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Inizializzazione del modello
model = BoundingBoxModel().to(device)

def smooth_l1_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + delta * linear

def diou_loss(y_true, y_pred):
    x_min_true, y_min_true, x_max_true, y_max_true = torch.chunk(y_true, 4, dim=-1)
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = torch.chunk(y_pred, 4, dim=-1)
    
    true_area = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    
    inter_x_min = torch.max(x_min_true, x_min_pred)
    inter_y_min = torch.max(y_min_true, y_min_pred)
    inter_x_max = torch.min(x_max_true, x_max_pred)
    inter_y_max = torch.min(y_max_true, y_max_pred)
    
    inter_width = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_height = torch.clamp(inter_y_max - inter_y_min, min=0)
    intersection = inter_width * inter_height
    
    union = true_area + pred_area - intersection
    iou = intersection / (union + 1e-7)
    
    true_center_x = (x_min_true + x_max_true) / 2.0
    true_center_y = (y_min_true + y_max_true) / 2.0
    pred_center_x = (x_min_pred + x_max_pred) / 2.0
    pred_center_y = (y_min_pred + y_max_pred) / 2.0
    
    center_distance = (true_center_x - pred_center_x) ** 2 + (true_center_y - pred_center_y) ** 2
    
    enclosing_x_min = torch.min(x_min_true, x_min_pred)
    enclosing_y_min = torch.min(y_min_true, y_min_pred)
    enclosing_x_max = torch.max(x_max_true, x_max_pred)
    enclosing_y_max = torch.max(y_max_true, y_max_pred)
    
    enclosing_diag = (enclosing_x_max - enclosing_x_min) ** 2 + (enclosing_y_max - enclosing_y_min) ** 2
    
    diou = iou - (center_distance / (enclosing_diag + 1e-7))
    return 1 - diou

def combined_loss(y_true, y_pred, alpha=0.08):
    smooth = smooth_l1_loss(y_true, y_pred).mean()
    diou = diou_loss(y_true, y_pred).mean()
    return alpha * smooth + (1 - alpha) * diou



# Metriche di valutazione
def mean_iou(y_true, y_pred):
    x_min_true, y_min_true, x_max_true, y_max_true = torch.chunk(y_true, 4, dim=-1)
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = torch.chunk(y_pred, 4, dim=-1)
    
    inter_x_min = torch.max(x_min_true, x_min_pred)
    inter_y_min = torch.max(y_min_true, y_min_pred)
    inter_x_max = torch.min(x_max_true, x_max_pred)
    inter_y_max = torch.min(y_max_true, y_max_pred)
    
    inter_width = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_height = torch.clamp(inter_y_max - inter_y_min, min=0)
    intersection = inter_width * inter_height
    
    true_area = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    
    union = true_area + pred_area - intersection
    iou = intersection / (union + 1e-7)
    return iou.mean()

def mean_center_distance(y_true, y_pred):
    x_center_true = (y_true[..., 0] + y_true[..., 2]) / 2.0
    y_center_true = (y_true[..., 1] + y_true[..., 3]) / 2.0
    
    x_center_pred = (y_pred[..., 0] + y_pred[..., 2]) / 2.0
    y_center_pred = (y_pred[..., 1] + y_pred[..., 3]) / 2.0
    
    distance = torch.sqrt((x_center_true - x_center_pred) ** 2 + (y_center_true - y_center_pred) ** 2)
    return distance.mean()


criterion = combined_loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.937, weight_decay=0.01)


# Training loop con salvataggio del modello e delle metriche
num_epochs = 80
losses = []
iou_scores = []
center_distances = []



for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_iou = 0
    total_distance = 0
    
    for images, bboxes in train_loader:
        images, bboxes = images.to(device), bboxes.to(device)
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        outputs = model(images)
        loss = combined_loss(outputs, bboxes)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += mean_iou(outputs, bboxes).item()
        total_distance += mean_center_distance(outputs, bboxes).item()
    
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_distance = total_distance / len(train_loader)
    
    losses.append(avg_loss)
    iou_scores.append(avg_iou)
    center_distances.append(avg_distance)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, IoU: {avg_iou}, Distance: {avg_distance}")


save_dir = "training_results"
os.makedirs(save_dir, exist_ok=True)

# Salvataggio del modello
model_path = os.path.join(save_dir, "model.pth")
torch.save(model.state_dict(), model_path)
print(f"Modello salvato in {model_path}")

# Generazione dei grafici
plt.figure()
plt.plot(range(1, num_epochs + 1), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, "loss_plot.png"))

plt.figure()
plt.plot(range(1, num_epochs + 1), iou_scores, label='Mean IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.savefig(os.path.join(save_dir, "iou_plot.png"))

plt.figure()
plt.plot(range(1, num_epochs + 1), center_distances, label='Mean Center Distance')
plt.xlabel('Epoch')
plt.ylabel('Distance')
plt.legend()
plt.savefig(os.path.join(save_dir, "distance_plot.png"))

print("Grafici salvati nella cartella training_results")