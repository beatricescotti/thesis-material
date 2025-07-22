import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom
import plotly.graph_objects as go
import os
import SimpleITK as sitk

# sistema il path della cartella in base a come lo hai in locale
dicom_folder = '/gwpool/users/bscotti/tesi/dati/set1//slices_filtrate_1'

dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]

# Usa SimpleITK per leggere la serie di immagini DICOM
reader = sitk.ImageSeriesReader()
reader.SetFileNames(dicom_files)
volume = reader.Execute()
volume_array = sitk.GetArrayFromImage(volume)  # Forma: (n_images, 512, 512)

# creazione della maschera in cui vengono selezionati solo i polmoni (grazie alle HU)
mask = volume_array < -320
mask = np.vectorize(clear_border, signature='(n,m)->(n,m)')(mask) # vectorize crea un loop su tutte le slices, e voglio il loop su immagini 2D
mask_labeled = np.vectorize(label, signature='(n,m)->(n,m)')(mask)

slc = mask_labeled[10]
rps = regionprops(slc)
areas = [r.area for r in rps]
idxs = np.argsort(areas)[::-1]

new_slc = np.zeros_like(slc)
for i in idxs[:3]:
  new_slc[tuple(rps[i].coords.T)] = i+1

def keep_top_3(slc):
    new_slc = np.zeros_like(slc)
    rps = regionprops(slc)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]
    for i in idxs[:3]:
        new_slc[tuple(rps[i].coords.T)] = i+1
    return new_slc

mask_labeled = np.vectorize(keep_top_3, signature='(n,m)->(n,m)')(mask_labeled)
mask = mask_labeled > 0
mask = np.vectorize(ndi.binary_fill_holes, signature='(n,m)->(n,m)')(mask)

def remove_trachea(slc, c= 0.0069):
    new_slc = slc.copy()
    labels = label(slc, connectivity=1, background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas/512**2 < c)[0]
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)]=0
    return new_slc
 

mask = np.vectorize(remove_trachea, signature = '(n,m)->(n,m)')(mask)
labels = label(mask[10], background=0)
center_of_mass(labels==3)[0]

def delete_table(slc):
    new_slc = slc.copy()
    labels = label(slc, background=0)
    idxs = np.unique(labels)[1:]
    COM_ys = np.array([center_of_mass(labels==i)[0] for i in idxs])
    for idx, COM_y in zip(idxs, COM_ys):
        if (COM_y < 0.3*slc.shape[0]):  # se il centro di massa è più grandel del 30%, calcellalo. quindi toglie tutta la parte sotto
            new_slc[labels==idx] = 0
        elif (COM_y > 0.6*slc.shape[0]):  # allo stesso tempo, se il centro di massa è maggiore del 60%, cancellalo! quindi toglie qualsiasi cosa sopra. lo faccio perchè a volte può esserci qualcosa sopra, ma in teoria non nel mio caso.
            new_slc[labels==idx] = 0
    return new_slc


mask_new = np.vectorize(delete_table, signature='(n,m)->(n,m)')(mask)
# img_new = mask_new * volume_array
img_new = mask_new * volume_array

output_folder = '/gwpool/users/bscotti/tesi/dati/set1/masked_slices1'
os.makedirs(output_folder, exist_ok=True)

for i, slc in enumerate(img_new):
    original_filename = os.path.basename(dicom_files[i])
    file_name_without_extension = os.path.splitext(original_filename)[0]
    output_filename = os.path.join(output_folder, f"{file_name_without_extension}.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(slc, cmap='gray')
    plt.axis('off')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()