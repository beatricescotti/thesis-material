import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os


array_npy = '/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_andre/2d_1_to_6_resampled'
array_paths = glob.glob(os.path.join(array_npy, '*.npy'))
for array_path in array_paths:
    array = np.load(array_path)

    def apply_window(hu_array, window_center, window_width):
        """
        hu_array: numpy array con valori in hounsfield units
        window_center: centro della finestra
        window_width: larghezza della finestra
        restituisce un array normalizzato tra 0 e 255
        """
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2
        windowed = np.clip(hu_array, window_min, window_max)
        normalized = ((windowed - window_min) / (window_max - window_min)) * 255
        return normalized.astype(np.uint8)

    RED_CENTER = -600
    RED_WIDTH = 1700

    GREEN_CENTER = -488
    GREEN_WIDTH = 1000 

    BLUE_CENTER =  -325
    BLUE_WIDTH =  700

    # applicazione delle finestre
    red_channel = apply_window(array, RED_CENTER, RED_WIDTH)   
    green_channel = apply_window(array, GREEN_CENTER, GREEN_WIDTH)    
    # green_channel = (green_channel/255)**0.4*255 # correzione gamma per il canale verde
    blue_channel = apply_window(array,  BLUE_CENTER, BLUE_WIDTH)    

    # combino i tre canali in un'immagine RGB
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    img = Image.fromarray(rgb_image)
    img_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_andre/2d_1_to_6_rgb'
    os.makedirs(img_path, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(array_path))[0] + '.png'
    img.save(os.path.join(img_path, img_name))
