import os
import glob
import numpy as np
import cv2

img_path = "/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/resampling"
img_paths = glob.glob(os.path.join(img_path, '*.png'))

target_size = (704, 704)  # (width, height)
padded_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/padded_slices_new'  
os.makedirs(padded_path, exist_ok=True)

for img_path in img_paths:
    # Carica l'immagine
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Legge l'immagine in scala di grigi
    h, w = img.shape
    print(f"Processing {img_path} - Original size: {h}x{w}")
    padded = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    # Assicuriamoci di non scrivere fuori dai bordi
    h_cropped = min(h, target_size[1])
    w_cropped = min(w, target_size[0])

    padded[0:h_cropped, 0:w_cropped] = img[0:h_cropped, 0:w_cropped]
    print(f"Padded size: {padded.shape}")

    padded_name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    cv2.imwrite(os.path.join(padded_path, padded_name), padded)
