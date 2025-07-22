import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path


input_folder = '/gwpool/users/bscotti/tesi/dati/final_jet/patches/P_NoBkg_001'
output_folder = '/gwpool/users/bscotti/tesi/dati/final_jet/patches/P_NoBkg_001_png'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        filepath = os.path.join(input_folder, filename)

        img_array = np.load(filepath)
        #img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
        #img_array = img_array.astype(np.uint8)

        #img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        img = Image.fromarray(img_array)  

        output_filename = os.path.splitext(filename)[0] + '.png'
        img.save(os.path.join(output_folder, output_filename))
print("immagini PNG sono state salvate in:", output_folder)