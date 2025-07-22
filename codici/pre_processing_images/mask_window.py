import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, center_of_mass
import os
import SimpleITK as sitk

# 1. Caricamento DICOM e preprocessing HU
dicom_folder = '/gwpool/users/bscotti/tesi/dati/dati_medici/set1/slices_filtrate_1'
dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]

reader = sitk.ImageSeriesReader()
reader.SetFileNames(dicom_files)
volume = reader.Execute()
hu_array = sitk.GetArrayFromImage(volume)  # Conserva i valori HU originali

# 2. Creazione maschera polmonare (usa HU grezzi)
mask = hu_array < -320  # Segmentazione su HU originali
mask = np.array([clear_border(slc) for slc in mask])
mask_labeled = np.array([label(slc) for slc in mask])

def keep_top_3(slc):
    new_slc = np.zeros_like(slc)
    rps = regionprops(slc)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]
    for i in idxs[:3]:
        new_slc[tuple(rps[i].coords.T)] = 1
    return new_slc

mask_labeled = np.array([keep_top_3(slc) for slc in mask_labeled])
mask_filled = np.array([binary_fill_holes(slc) for slc in mask_labeled])

# 3. Pulizia maschera (trachea/tavolo)
def remove_trachea(slc, c=0.0069):
    new_slc = slc.copy()
    labels = label(slc, connectivity=1, background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas / 512**2 < c)[0]
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)] = 0
    return new_slc

mask_clean = np.array([remove_trachea(slc) for slc in mask_filled])

def delete_table(slc):
    new_slc = slc.copy()
    labels = label(slc, background=0)
    idxs = np.unique(labels)[1:]
    COM_ys = np.array([center_of_mass(labels == i)[0] for i in idxs])
    for idx, COM_y in zip(idxs, COM_ys):
        if COM_y < 0.3 * slc.shape[0] or COM_y > 0.6 * slc.shape[0]:
            new_slc[labels == idx] = 0
    return new_slc

final_mask = np.array([delete_table(slc) for slc in mask_clean])

# 4. Applica windowing SOLO alle aree polmonari
window_center = -600
window_width = 1500
window_min = window_center - window_width // 2
window_max = window_center + window_width // 2

# Crea immagine finale (sfondo bianco)
final_img = np.full_like(hu_array, 255, dtype=np.uint8)  # Sfondo bianco

for i in range(len(hu_array)):
    # Applica windowing solo dove la maschera è True
    hu_slice = hu_array[i]
    mask_slice = final_mask[i]
    
    # Clip e normalizza
    hu_windowed = np.clip(hu_slice[mask_slice], window_min, window_max)
    hu_normalized = 255 * (hu_windowed - window_min) / window_width
    final_img[i][mask_slice] = hu_normalized.astype(np.uint8)

# 5. Salvataggio
output_folder = '/gwpool/users/bscotti/tesi/dati/dati_medici/NLST/prova_mask_window'
os.makedirs(output_folder, exist_ok=True)

for i, slc in enumerate(final_img):
    original_filename = os.path.basename(dicom_files[i])
    output_filename = os.path.join(output_folder, f"{os.path.splitext(original_filename)[0]}.png")
    plt.imsave(output_filename, slc, cmap='gray', vmin=0, vmax=255)
    print(f"Salvata: {output_filename}")