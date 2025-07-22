import numpy as np
from PIL import Image
import os
import glob

# faccio un ciclo su tutte le immagini della cartella 2d_1_to_6_resampled

array_npy = '/gwdata/users/aborghesi/NLST/resampled_slices'
array_paths = glob.glob(os.path.join(array_npy, '*.npy'))
for array_path in array_paths:
    array = np.load(array_path)

    # definizione della window, parametri per mettere in risalto i polmoni
    window_center = -600    
    window_width = 1700   
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    array_windowed = np.clip(array, window_min, window_max)  
    array_normalized = 255 * (array_windowed - window_min) / (window_width)
    array_uint8 = array_normalized.astype(np.uint8) # conversione da float32 a unit8

    img = Image.fromarray(array_uint8)
    img_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/resampled_windowing'
    os.makedirs(img_path, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(array_path))[0] + '.png'
    img.save(os.path.join(img_path, img_name))
