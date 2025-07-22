import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, center_of_mass
import os
import SimpleITK as sitk

# Imposta il path della cartella DICOM
dicom_folder = '/gwpool/users/bscotti/tesi/dati/set1/slices_filtrate_1'
dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]

# Usa SimpleITK per leggere la serie di immagini DICOM
reader = sitk.ImageSeriesReader()
reader.SetFileNames(dicom_files)
volume = reader.Execute()
volume_array = sitk.GetArrayFromImage(volume)  # Forma: (n_images, 512, 512)

# Creazione della maschera per selezionare i polmoni (grazie alle HU)
mask = volume_array < -320  # Consideriamo i polmoni HU < -320
mask = np.array([clear_border(slc) for slc in mask])  # Rimuove eventuali connessioni ai bordi
mask_labeled = np.array([label(slc) for slc in mask])  # Etichettiamo le regioni con numeri distinti

# Funzione per mantenere le 3 regioni più grandi (polmoni)
def keep_top_3(slc):
    new_slc = np.zeros_like(slc)
    rps = regionprops(slc)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]  # Ordina per area decrescente
    for i in idxs[:3]:  # Prendiamo le 3 più grandi
        new_slc[tuple(rps[i].coords.T)] = 1
    return new_slc

mask_labeled = np.array([keep_top_3(slc) for slc in mask_labeled])
mask = np.array([binary_fill_holes(slc) for slc in mask_labeled])  # Riempiamo i buchi interni

# Funzione per rimuovere la trachea (se presente)
def remove_trachea(slc, c=0.0069):
    new_slc = slc.copy()
    labels = label(slc, connectivity=1, background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas / 512**2 < c)[0]  # Se troppo piccola, consideriamo trachea
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)] = 0
    return new_slc

mask = np.array([remove_trachea(slc) for slc in mask])

# Funzione per rimuovere il tavolo (sostanze estranee)
def delete_table(slc):
    new_slc = slc.copy()
    labels = label(slc, background=0)
    idxs = np.unique(labels)[1:]  # Prendiamo le regioni etichettate
    COM_ys = np.array([center_of_mass(labels == i)[0] for i in idxs])  # Centroide delle regioni
    for idx, COM_y in zip(idxs, COM_ys):
        if COM_y < 0.3 * slc.shape[0] or COM_y > 0.6 * slc.shape[0]:  # Fuori dalla zona polmonare
            new_slc[labels == idx] = 0
    return new_slc

mask_new = np.array([delete_table(slc) for slc in mask])

# **NORMALIZZAZIONE E SFONDO BIANCO**
global_min = -1000  # Valore minimo fisso per normalizzazione
global_max = 0      # Valore massimo fisso

# Inizializziamo un array completamente bianco
normalized_img = np.ones_like(volume_array, dtype=np.float32)

# Normalizzazione dei polmoni nel range [0,1]
normalized_values = (volume_array - global_min) / (global_max - global_min)
normalized_values = np.clip(normalized_values, 0, 1)  # Assicura che i valori siano tra 0 e 1

# Applichiamo solo ai polmoni
normalized_img[mask_new] = normalized_values[mask_new]

# **Salvataggio delle immagini**
output_folder = '/gwpool/users/bscotti/tesi/dati/slices_white'
os.makedirs(output_folder, exist_ok=True)

for i, slc in enumerate(normalized_img):
    original_filename = os.path.basename(dicom_files[i])
    file_name_without_extension = os.path.splitext(original_filename)[0]
    output_filename = os.path.join(output_folder, f"{file_name_without_extension}.png")

    plt.imsave(output_filename, slc, cmap='gray')  # Salva in scala di grigi
    print(f"Salvata: {output_filename} - Dimensioni: {slc.shape}")