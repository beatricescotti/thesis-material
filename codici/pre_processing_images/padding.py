import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os


img_path = "/gwdata/users/aborghesi/NLST/resampled_slices"
img_paths = glob.glob(os.path.join(img_path, '*.png'))

for img_path in img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Legge l'immagine in scala di grigi

    target_size = (704, 704) 
    padded = np.zeros((target_size[1], target_size[0], target_size[2]), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    start_x = 0
    start_y = 0 
    end_x = w
    end_y = h
    padded[start_y:end_y, start_x:end_x] = img

    padded_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/padded_slices'  
    os.makedirs(padded_path, exist_ok=True)
    padded_name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    cv2.imwrite(os.path.join(padded_path, padded_name), padded)