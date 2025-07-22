import os
import shutil
import random

# Cartella di origine (dove sono le immagini)
source_folder = '/gwpool/users/bscotti/tesi/dati/set2/slices_filtrate_2'

# Cartella di destinazione (dove copiare le immagini)
destination_folder = '/gwpool/users/bscotti/tesi/dati/prova'
os.makedirs(destination_folder, exist_ok=True)

# Ottieni la lista di tutte le immagini (filtra per estensione se necessario)
images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.dcm'))]

# Se ci sono meno di 50 immagini, prendile tutte
num_images_to_copy = min(60, len(images))

# Scegli 50 immagini a caso
selected_images = random.sample(images, num_images_to_copy)

# Copia le immagini nella cartella di destinazione
for image in selected_images:
    shutil.copy(os.path.join(source_folder, image), os.path.join(destination_folder, image))

print(f"Copiate {num_images_to_copy} immagini in {destination_folder}")