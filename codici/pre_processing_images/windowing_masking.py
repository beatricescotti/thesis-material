import numpy as np
from PIL import Image
import os
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, disk 
import glob 


array_npy = '/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_andre/2d_1_to_6_resampled'
array_paths = glob.glob(os.path.join(array_npy, '*.npy'))
for array_path in array_paths:
    array = np.load(array_path)

    # Windowing e normalizzazione
    window_center = -600    
    window_width = 1700   
    window_min = window_center - window_width // 2  # -1450 HU
    window_max = window_center + window_width // 2  # +250 HU

    array_windowed = np.clip(array, window_min, window_max)  
    array_normalized = 255 * (array_windowed - window_min) / window_width
    array_uint8 = array_normalized.astype(np.uint8)

    # creazione della maschera
    mask_polmoni = (array >= -1000) & (array <= -400)

    # pulizia della maschera
    mask_polmoni = clear_border(mask_polmoni)
    mask_polmoni = binary_opening(mask_polmoni, disk(2))  

    # riempimento dei buchi nel polmone
    mask_polmoni = binary_fill_holes(mask_polmoni).astype(np.uint8) 

    # Selezione delle 2 regioni più grandi (polmoni)
    labeled_mask = label(mask_polmoni)
    regions = regionprops(labeled_mask)
    if len(regions) > 0:
        regions_sorted = sorted(regions, key=lambda x: x.area, reverse=True)[:2]
        final_mask = np.zeros_like(mask_polmoni)
        for region in regions_sorted:
            final_mask[tuple(region.coords.T)] = 1
    else:
        final_mask = mask_polmoni  # Fallback se non trova regioni

    ### 3. Rimozione trachea (regione piccola in alto) 
    for region in regionprops(label(final_mask)):
        cy, _ = region.centroid
        if cy < final_mask.shape[0] * 0.25 and region.area < 500:  # Soglie personalizzabili
            final_mask[tuple(region.coords.T)] = 0

    # applicazione della maschera
    array_masked = array_windowed * final_mask
    array_masked_normalized = 255 * (array_masked - window_min) / window_width
    array_masked_uint8 = np.clip(array_masked_normalized, 0, 255).astype(np.uint8)


    background = (final_mask == 0)
    array_masked_uint8[background] = 255  
    array_rgb = np.stack([array_masked_uint8]*3, axis=-1) 


    img_path = '/gwpool/users/bscotti/tesi/dati/dati_medici/dataset_andre/2d_1_to_6_png_masked'
    os.makedirs(img_path, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(array_path))[0] + '.png'
    Image.fromarray(array_rgb).save(os.path.join(img_path, img_name))



