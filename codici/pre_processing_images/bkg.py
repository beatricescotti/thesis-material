import cv2
import numpy as np
import os

# Cartelle di input e output
input_folder = "/gwpool/users/bscotti/tesi/dati/dataset0/images/test"  # Cambia con il percorso della cartella sorgente
output_folder = "/gwpool/users/bscotti/tesi/dati/dataset0/images/test_new"  # Cambia con il percorso della cartella di destinazione
os.makedirs(output_folder, exist_ok=True)  # Crea la cartella di output se non esiste

# Processa ogni immagine nella cartella di input
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Filtra solo le immagini
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            # Trova il pixel più frequente
            unique, counts = np.unique(image, return_counts=True)
            most_common_pixel = unique[np.argmax(counts)]

            # Sostituisci il pixel più frequente con 255
            image_processed = np.where(image == most_common_pixel, 255, image).astype(np.uint8)

            # Salva l'immagine processata
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image_processed)
            print(f"Salvata: {output_path}")

print("Elaborazione completata!")
